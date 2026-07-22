//! Elementwise compute operations.

use hephaestus_core::{HephaestusError, Result};

use crate::application::pipeline::encode_compute_pass;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Binary elementwise compute operations.
pub mod binary;
/// Scalar elementwise compute operations.
pub mod scalar;
/// Unary elementwise compute operations.
pub mod unary;

pub use binary::{AddOp, DivOp, MulOp, PowOp, SubOp, binary_elementwise, binary_elementwise_into};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    AbsOp, CosOp, ExpNegOp, ExpOp, IdentityOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp,
    unary_elementwise, unary_elementwise_into,
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
    encode_compute_pass(&mut encoder, pipeline, &bind_group, groups, label);
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
