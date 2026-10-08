//! Elementwise compute operations.

use hephaestus_core::{DialectScalar, HephaestusError, Result, UnaryExpr, Wgsl};

use crate::application::pipeline::encode_compute_pass;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Binary elementwise compute operations.
pub mod binary;
/// Scalar elementwise compute operations.
pub mod scalar;
/// Unary elementwise compute operations.
pub mod unary;

pub use binary::{
    AddOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, PowOp, SubOp, binary_elementwise,
    binary_elementwise_into, binary_elementwise_typed, binary_elementwise_typed_into,
};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, EluGradOp,
    EluOp, ErfOp, ErfcOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp, GeluGradOp, GeluOp,
    GeluTanhGradOp, GeluTanhOp, HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp,
    IdentityOp, LgammaOp, LnOp, Log1pOp, Log2Op, Log10Op, MishGradOp, MishOp, NegOp, RecipOp,
    ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp, SiluGradOp, SiluOp, SinOp,
    SincOp, SinhOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, SqrtOp, TanOp,
    TanhGradOp, TanhOp, TruncOp, unary_elementwise, unary_elementwise_into,
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

/// Reject a unary dispatch whose WGSL template cannot target `f64` buffers.
///
/// Templates reporting [`UnaryExpr::SUPPORTS_F64`] `false` spell `select`
/// with literal-only arms, which concretize to `f32` and fail shader
/// validation against `array<f64>` outputs. Failing here keeps the error a
/// typed unsupported-operation rejection instead of a driver compile
/// failure. Every unary shader-build path (contiguous, strided, seam
/// prepare) calls this before emitting WGSL.
pub(crate) fn reject_f64_unsupported_unary<Op, T>() -> Result<()>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl>,
{
    if !Op::SUPPORTS_F64 && core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
        return Err(HephaestusError::Unsupported {
            message: format!(
                "{} does not support f64 in WGSL: literal-only `select` arms concretize f32",
                core::any::type_name::<Op>()
            ),
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
