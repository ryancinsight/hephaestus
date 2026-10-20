//! Accuracy sweep for [`SoftplusOp`](hephaestus_core::SoftplusOp).
//!
//! The renderings are the two-sided stabilized form: the C dialects branch
//! on the sign around native `log1p` (`x <= 0 ? log1p(exp(x)) :
//! x + log1p(exp(-x))`), while WGSL evaluates `log1p(exp(-|x|))` through
//! two bands of the shared `log1p` core (the core band plus the exact upper
//! shift; the outer `log` bands are unreachable over `exp` of a
//! non-positive argument). The sweep covers the overflow knee past
//! `ln(FLT_MAX)`, the underflow tail where `1 + exp(x)` rounds to `1`, the
//! transition interior, and the structural specials.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, SoftplusOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// 4096 points over `[-10, 10]`, 512 over the positive tail, plus the
/// overflow knee, the underflow knee, and the structural specials:
/// everything at or above the driver-`exp` tail band, swept at the form
/// bound. The positive tail sweeps here rather than in the tail band
/// because its error vanishes (the result nears `x`, so the `log1p`
/// perturbation is sub-ULP).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -10.0,
        10.0,
        4096,
        &[
            88.72, // just below ln(FLT_MAX): finite
            88.73, // just past it: the value stays finite
            89.0, 100.0, // overflow interior: the value is ~x
            -10.0, // tail-band handoff (also covered by the tail sweep)
            -1.0, 1.0, // curvature interior
            0.0, // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(10.0, 110.0, 512, &[]));
    inputs
}

/// 512 points over the negative tail band `[-110, -10)`, swept at the
/// driver-`exp` bound: the rendering is transparent `log1p(exp(x))` with
/// `exp(x)` tiny there, so the error is the driver's `exp` grade at large
/// arguments, which the Vulkan driver spends loosely — the same loose
/// grade the `expm1` knee band records on the positive side.
fn tail_inputs() -> Vec<f32> {
    sweep_inputs(
        -110.0,
        -10.0,
        512,
        &[
            -10.0,  // tail-band handoff (also covered by the main sweep)
            -16.6,  // 1 + exp(x) rounds to 1 below here
            -87.0,  // WGSL flush floor: the last swept subnormal-free input
            -88.72, // just above the exp knee (subnormal result: CUDA/HIP only)
            -88.73, // just past it: exp overflows under the old form
            -89.0, -100.0, // saturation interior
        ],
    )
}

/// Sweep [`SoftplusOp`](hephaestus_core::SoftplusOp) on one backend.
///
/// Bounds: CUDA `4` ULP (`log1pf` at `1`, `expf` at `2` with unit input
/// sensitivity, `+` at `0.5`, plus margin); HIP `4` ULP (`log1pf` at `1`,
/// `expf` at `1` on its tested range, `+` at `0.5`, plus margin); WGSL `16`
/// ULP (the two-band `log1p` rendering at its `6`-ULP bound over the
/// driver `exp` at arguments to `-10` — lavapipe measures `9.9` total
/// worst at `x = -7.68`, the `exp` term running `≈ 3.4` there against
/// the canary `1.7`, inside the `(3+2|x|)` envelope — with unit input
/// sensitivity, `+` at `0.5`, plus margin for cross-driver variance).
/// WGSL additionally drops `x < -87`, where the true result is
/// subnormal and flushing is permitted; CUDA and HIP sweep the tail as
/// IEEE.
/// The negative tail band `[-110, -10)` sweeps separately at the
/// driver-`exp` bound, since the rendering is transparent there: CUDA `4`
/// and HIP `4` ULP (the native builtin holds its row across the band),
/// WGSL `64` ULP (the Vulkan driver at 610.60 measures 57.3 ULP worst at
/// `x = -75.6`, the same loose large-argument `exp` grade the `sigmoid`
/// tail and the `expm1` knee record — and the canary's `exp` probes cover
/// only their eight sample points, so the sweep carries the grade instead;
/// see the canary-coverage note in `mod.rs`). The split sits at `-10`:
/// the main bound holds over the dense interior as measured, while a
/// provisional split at `-20` failed the main band at 11.4 ULP
/// (`x = -17.48`).
pub fn assert_softplus_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SoftplusOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, tail_ulp) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (4.0, 4.0),
        SweepDialect::Wgsl => (16.0, 64.0),
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
    assert_unary_sweep_within::<D, E, SoftplusOp>(device, ops, "softplus", &main, max_ulp);
    assert_unary_sweep_within::<D, E, SoftplusOp>(device, ops, "softplus-tail", &tail, tail_ulp);
}
