//! Shared WGPU pipeline and dispatch utilities.

use hephaestus_core::{BlockWidth, HephaestusError, Result};
use mnemosyne_core::KernelResourceBudget;

use crate::infrastructure::device::{PipelineKey, WgpuDevice};

/// Fetch the cached pipeline for `key`, compiling `source` on first use.
#[must_use]
pub(crate) fn cached_pipeline(
    device: &WgpuDevice,
    key: PipelineKey,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> wgpu::ComputePipeline {
    {
        let cache = device
            .pipeline_cache
            .lock()
            .expect("invariant: pipeline cache mutex is not poisoned");
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
    }

    let module = device
        .inner()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
    let pipeline = device
        .inner()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut cache = device
        .pipeline_cache
        .lock()
        .expect("invariant: pipeline cache mutex is not poisoned");
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }
    cache.insert(key, pipeline.clone());
    pipeline
}

/// Convert a logical work-item count into WGPU workgroup count.
pub(crate) fn workgroups(len: usize, width: BlockWidth) -> Result<u32> {
    let len = u64::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("dispatch size {len} exceeds u64 range"),
    })?;
    let checked =
        width
            .checked_covering_blocks(len)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("dispatch size {len} exceeds u32 workgroup range"),
            })?;
    let budget = KernelResourceBudget::new(0, 0, width.get())
        .expect("invariant: BlockWidth is non-zero, so budget threads are non-zero");
    let planned = moirai_gpu::plan_launch(budget, len);
    debug_assert_eq!(planned.threads_per_block, width.get());
    debug_assert_eq!(planned.grid_blocks, checked);
    Ok(planned.grid_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroups_accepts_exact_u32_group_limit() {
        let width = BlockWidth::new(256).expect("invariant: test width is non-zero");
        let len = u64::from(width.get()) * u64::from(u32::MAX);
        match workgroups(len as usize, width) {
            Ok(groups) => assert_eq!(groups, u32::MAX),
            Err(error) => panic!("expected max workgroup count, got {error:?}"),
        }
    }

    #[test]
    fn workgroups_rejects_beyond_u32_group_limit() {
        let width = BlockWidth::new(256).expect("invariant: test width is non-zero");
        let len = u64::from(width.get()) * u64::from(u32::MAX) + 1;
        match workgroups(len as usize, width) {
            Err(HephaestusError::DispatchFailed { message }) => assert_eq!(
                message,
                format!("dispatch size {len} exceeds u32 workgroup range")
            ),
            other => panic!("expected dispatch-size rejection, got {other:?}"),
        }
    }
}
