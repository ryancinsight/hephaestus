//! Elementwise compute operations.

use hephaestus_core::{HephaestusError, Result};

use crate::application::pipeline::workgroups;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Binary elementwise compute operations.
pub mod binary;
/// Scalar elementwise compute operations.
pub mod scalar;
/// Unary elementwise compute operations.
pub mod unary;

pub use binary::{
    binary_elementwise, binary_elementwise_into, AddOp, BinaryWgslOp, DivOp, MulOp, PowOp, SubOp,
};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    unary_elementwise, unary_elementwise_into, AbsOp, CosOp, ExpOp, IdentityOp, LnOp, NegOp,
    RecipOp, SinOp, SqrtOp, UnaryWgslOp,
};

fn reject_output_alias<T, U>(
    input_label: &'static str,
    input: &WgpuBuffer<T>,
    out: &WgpuBuffer<U>,
) -> Result<()> {
    if input.aliases(out) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("output buffer must not alias {input_label} input"),
        });
    }
    Ok(())
}

/// Encode a single-pass elementwise compute dispatch.
///
/// This is the SSOT for the encode-bind-dispatch pattern shared by
/// [`binary_elementwise_into`], [`unary_elementwise_into`], and
/// [`scalar_elementwise_into`]. Callers build their `entries` slice on the
/// stack (max 3 entries) and pass the already-computed workgroup count.
///
/// # Errors
///
/// Returns `DispatchFailed` if the workgroup count computation overflows
/// `u32`.
pub(crate) fn encode_elementwise(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    label: &'static str,
    entries: &[wgpu::BindGroupEntry<'_>],
    groups: u32,
) -> Result<()> {
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

/// Compute the workgroup count and reject empty inputs.
///
/// Returns `None` when `len == 0` (caller should return `Ok(())` immediately).
#[allow(dead_code)]
pub(crate) fn elementwise_groups(
    len: usize,
    width: hephaestus_core::BlockWidth,
) -> Result<Option<u32>> {
    if len == 0 {
        return Ok(None);
    }
    workgroups(len, width).map(Some)
}
