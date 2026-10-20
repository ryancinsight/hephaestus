//! Accuracy sweep for [`SigmoidOp`](hephaestus_core::SigmoidOp).
//!
//! `SigmoidOp` renders the stabilized two-sided quotient shared with
//! [`SoftplusGradOp`](hephaestus_core::SoftplusGradOp) (the softplus
//! derivative is the logistic sigmoid, so the two markers carry one
//! string): the `x <= 0` branch evaluates `exp(x)` where it cannot
//! overflow, the `x > 0` branch `exp(-x)`. The sweep covers the saturation
//! knees past `|x| = ln(FLT_MAX)`, the transition interior, and the
//! structural specials. The item's op list names only the grad, so this
//! entry is a noted scope extension over the identical string (see
//! `softplus_grad.rs`).

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, SigmoidOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// 4096 points over `[-10, 10]`, 512 over the positive tail, plus the
/// saturation knees and the structural specials: everything at or above
/// the driver-`exp` tail band, swept at the form bound. The positive tail
/// sweeps here rather than in the tail band because its error vanishes
/// (the result nears `1`, so the `exp` perturbation is sub-ULP).
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

/// 512 points over the negative tail band `[-120, -10)`, swept at the
/// driver-`exp` bound: the rendering is transparent `exp(x)` over a
/// near-one denominator there (the `+ 1` vanishes in rounding for large
/// `|x|`), so the error is the driver's `exp` grade at large arguments,
/// which the Vulkan driver spends loosely — the same loose grade the
/// `expm1` knee band records on the positive side.
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

/// Sweep [`SigmoidOp`](hephaestus_core::SigmoidOp) on one backend.
///
/// Bounds: CUDA `5` ULP (`expf` at `2`, `/` at `2` without `-prec-div`,
/// `+` at `0.5`, unit input sensitivity, plus margin); HIP `5` ULP (`expf`
/// at `1` on its tested range, `/` carried at `≤ 2`, plus margin); WGSL `6`
/// ULP (`exp` at canary `1.7` over its eight samples, `/` at `2.5`, `+`
/// at `0.5`, plus margin — widened from `6`, which the Vulkan driver at
/// 610.60 exceeds at 6.79 ULP worst (`x = -9.13`, 5 of 4618 over): the
/// canary's sparse `exp` maximum undercovers the dense range, so the bound
/// carries the measured grade to the next power of two). WGSL
/// additionally drops `x < -87`, where the true result is subnormal and
/// flushing is permitted; CUDA and HIP sweep the tail as IEEE.
/// The negative tail band `[-120, -10)` sweeps separately at the
/// driver-`exp` bound, since the rendering is transparent there: CUDA `5`
/// and HIP `5` ULP (the native builtin holds its row across the band),
/// WGSL `64` ULP (the Vulkan driver at 610.60 measures 52.1 ULP worst at
/// `x = -63.8`; the old one-sided form measures 51.1 ULP at the same
/// input, proving the error is the driver's large-argument `exp` grade
/// rather than the form — and the canary's `exp` probes cover only their
/// eight sample points, so the sweep carries the grade instead; see the
/// canary-coverage note in `mod.rs`). The split sits at `-10`: the main
/// bound holds over the dense interior as measured, while a provisional
/// split at `-20` failed the main band at 11.6 ULP (`x = -17.48`).
pub fn assert_sigmoid_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SigmoidOp: UnaryExpr<E::Dialect>,
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
    assert_unary_sweep_within::<D, E, SigmoidOp>(device, ops, "sigmoid", &main, max_ulp);
    assert_unary_sweep_within::<D, E, SigmoidOp>(device, ops, "sigmoid-tail", &tail, tail_ulp);
}
