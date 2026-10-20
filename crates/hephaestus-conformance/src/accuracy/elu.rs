//! Accuracy sweep for [`EluOp`](hephaestus_core::EluOp).
//!
//! The positive branch is the identity (exact); the failing range is the
//! negative branch's `exp(x) - 1`, the same cancellation class as
//! [`Expm1Op`](hephaestus_core::Expm1Op), swept with that operator's
//! coverage over `x < 0` plus the kink at zero.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, EluOp, UnaryExpr};

use super::{SweepDialect, assert_unary_sweep_within, sweep_inputs, without_subnormals};

/// 4096 points over `[-1, 0]` plus the negative tail, the kink, and the
/// positive identity interior.
fn inputs() -> Vec<f32> {
    sweep_inputs(
        -1.0,
        0.0,
        4096,
        &[
            0.0, // +0 kink (the builder adds -0)
            1.0e-8,
            -1.0e-8, // cancellation at the kink
            2.0f32.powi(-24),
            -2.0f32.powi(-24), //
            -0.3,              // Taylor/direct handoff on the negative branch
            1.0,
            10.0,
            100.0, // positive identity interior
            -10.0,
            -87.0,
            -88.0, // negative tail
            -100.0,
            -104.0,
            -1000.0, // exp vanishes: the branch is -1
            -1.0e30, // deep tail
            1.0e-40,
            -1.0e-40, // subnormal inputs
        ],
    )
}

/// Sweep [`EluOp`](hephaestus_core::EluOp) on one backend.
///
/// Bounds match the `expm1` family (the negative branch renders that form;
/// the positive branch and the kink select are exact): CUDA `1.5` ULP, HIP
/// `1` ULP, WGSL `6` ULP. See `expm1.rs` for the derivations, including the
/// measured `expm1f` exceedance the CUDA bound absorbs (the ELU sweep
/// measures the same class at 1.075 ULP).
pub fn assert_elu_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    EluOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let max_ulp = match dialect {
        SweepDialect::CudaC => 1.5,
        SweepDialect::HipC => 1.0,
        SweepDialect::Wgsl => 6.0,
    };
    let inputs = inputs();
    // WGSL permits flushing subnormals; the other dialects sweep them.
    let inputs = if dialect == SweepDialect::Wgsl {
        without_subnormals(&inputs)
    } else {
        inputs
    };
    assert_unary_sweep_within::<D, E, EluOp>(device, ops, "elu", &inputs, max_ulp);
}
