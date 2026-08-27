//! Shared validation and submission for prepared WGPU operations.

use hephaestus_core::{HephaestusError, Result};

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

#[inline]
pub(crate) fn device_owner(device: &WgpuDevice) -> PipelineCache {
    device.pipeline_cache.clone()
}

pub(crate) fn validate_device_owner(
    owner: &PipelineCache,
    device: &WgpuDevice,
    operation: &str,
) -> Result<()> {
    if std::sync::Arc::ptr_eq(owner, &device.pipeline_cache) {
        Ok(())
    } else {
        Err(HephaestusError::DispatchFailed {
            message: format!("prepared WGPU {operation} belongs to a different device"),
        })
    }
}

pub(crate) fn validate_buffer_owner<T>(
    buffer: &WgpuBuffer<T>,
    device: &WgpuDevice,
    operation: &str,
) -> Result<()> {
    if buffer.belongs_to(&device.pipeline_cache) {
        Ok(())
    } else {
        Err(HephaestusError::DispatchFailed {
            message: format!("WGPU {operation} buffer belongs to a different device"),
        })
    }
}

pub(crate) fn checked_bind_group(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    label: &'static str,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> Result<wgpu::BindGroup> {
    let validation = device
        .inner()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });
    if let Some(error) = moirai::block_on(validation.pop()) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{label} bind-group creation failed: {error}"),
        });
    }
    Ok(bind_group)
}

pub(crate) fn checked_submit(
    device: &WgpuDevice,
    label: &'static str,
    encoder: wgpu::CommandEncoder,
) -> Result<()> {
    checked_submit_with_timeout(device, label, encoder, None).map(|_| ())
}

/// Submit `encoder` under scoped error detection, returning the submission's
/// queue index so callers can wait on exactly this submission instead of
/// draining the whole queue.
pub(crate) fn checked_submit_with_timeout(
    device: &WgpuDevice,
    label: &'static str,
    encoder: wgpu::CommandEncoder,
    timeout: Option<std::time::Duration>,
) -> Result<wgpu::SubmissionIndex> {
    let out_of_memory = device
        .inner()
        .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.inner().push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device
        .inner()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let submission_index = device.queue().submit(Some(encoder.finish()));
    let validation = validation.pop();
    let internal = internal.pop();
    let out_of_memory = out_of_memory.pop();

    if let Some(timeout) = timeout {
        device
            .inner()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index.clone()),
                timeout: Some(timeout),
            })
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("{label} submission wait failed: {error}"),
            })?;
    }

    // The bounded path has already completed the indexed submission and its
    // error-scope callbacks, so these executor calls cannot extend its deadline.
    if let Some(error) = [
        moirai::block_on(validation),
        moirai::block_on(internal),
        moirai::block_on(out_of_memory),
    ]
    .into_iter()
    .flatten()
    .next()
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{label} submission failed: {error}"),
        });
    }
    Ok(submission_index)
}
