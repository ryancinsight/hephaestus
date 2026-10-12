//! Accuracy sweep for [`SoftplusGradOp`](hephaestus_core::SoftplusGradOp).
//!
//! The softplus derivative is the logistic sigmoid, rendered (like
//! [`SigmoidOp`](hephaestus_core::SigmoidOp)) as the stabilized two-sided
//! quotient: the `x <= 0` branch evaluates `exp(x)` where it cannot
//! overflow, the `x > 0` branch `exp(-x)`. Same string, same sweep shape
//! as `sigmoid.rs`.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, SoftplusGradOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// Same main band as `sigmoid.rs`: 4096 points over `[-10, 10]`, 512
/// over the positive tail, plus the saturation knees and the structural
/// specials.
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -10.0,
        10.0,
        4096,
        &[
            88.72, 88.73, 89.0, 100.0, // positive saturation interior
            -10.0, // tail-band handoff (also covered by the tail sweep)
            0.0,   // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(10.0, 120.0, 512, &[]));
    inputs
}

/// Same negative tail band as `sigmoid.rs`: 512 points over `[-120, -10)`,
/// swept at the driver-`exp` bound.
fn tail_inputs() -> Vec<f32> {
    sweep_inputs(
        -120.0,
        -10.0,
        512,
        &[
            -10.0,  // tail-band handoff (also covered by the main sweep)
            -87.0,  // WGSL flush floor: the last swept subnormal-free input
            -88.72, // just above the exp knee (subnormal result: CUDA/HIP only)
            -88.73, // just past it: exp overflows under the old form
            -89.0, -100.0, // saturation interior
        ],
    )
}

/// Sweep [`SoftplusGradOp`](hephaestus_core::SoftplusGradOp) on one backend.
///
/// Bounds match the stabilized sigmoid composition: CUDA `5` ULP, HIP `5`
/// ULP, WGSL `8` ULP on the main band and `64` ULP on the negative tail
/// band. See `sigmoid.rs` for the derivations. WGSL drops `x < -87`
/// (subnormal results); CUDA and HIP sweep the tail as IEEE.
pub fn assert_softplus_grad_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SoftplusGradOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, tail_ulp) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (5.0, 5.0),
        SweepDialect::Wgsl => (8.0, 64.0),
    };
    let main = main_inputs();
    let tail = tail_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, tail) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -87.0),
            above_flush_floor(&without_subnormals(&tail), -87.0),
        )
    } else {
        (main, tail)
    };
    assert_unary_sweep_within::<D, E, SoftplusGradOp>(device, ops, "softplus-grad", &main, max_ulp);
    assert_unary_sweep_within::<D, E, SoftplusGradOp>(
        device,
        ops,
        "softplus-grad-tail",
        &tail,
        tail_ulp,
    );
}
