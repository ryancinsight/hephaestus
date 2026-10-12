//! Accuracy sweep for [`GeluTanhOp`](hephaestus_core::GeluTanhOp).
//!
//! The renderings apply the `0.5·(1 + tanh(z)) = sigmoid(2z)` identity to
//! the textbook `0.5·x·(1 + tanh(z))` (`z = √(2/π)·(x + 0.044715·x³)`),
//! uniformly: `x·sigmoid(2z)` with the stabilized two-sided sigmoid, so no
//! rendering forms `1 + tanh(z)` (which rounds to `0` for `x <= -5.5`
//! while the value stays a normal float down to `x ≈ -10.2`) or calls
//! `tanh` at all. `x = -∞` guards to `-0` (the unguarded form makes
//! `-∞·0`). The sweep covers the old-form zero crossing, the cubic-driven
//! tail where the value dies super-exponentially, and the structural
//! specials. The negative side grades by `|x|³` conditioning: the `2z`
//! argument rounds in absolute terms while the sigmoid transfers
//! absolute argument error 1:1 into relative output error, so the error
//! grows with `|z|` down the tail (`≈ 3.2·|z|` ULP calibrated at
//! `x = -5.73`, consistent with the rounding analysis).

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, GeluTanhOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// 4096 points over `[-2, 4]`, 256 over the positive tail, plus the
/// structural specials: everything at or above the conditioning mid
/// band, swept at the form bound. The positive tail sweeps here because
/// its error vanishes (the sigmoid saturates to exactly `1`, so the
/// result is `x`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -2.0,
        4.0,
        4096,
        &[
            -2.0, // mid-band handoff (also covered by the mid sweep)
            0.0,  // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(4.0, 20.0, 256, &[]));
    inputs
}

/// 512 points over the negative mid band `[-6, -2)`, swept at the
/// conditioning bound: `|z|` runs `1.9` to `12.5` here, so the envelope
/// runs about `12` to `46` ULP. Covers the old-form zero crossing at
/// `x = -5.5` (the ADR case: `-0` versus `-5.93e-9`).
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -6.0,
        -2.0,
        512,
        &[
            -6.0, // far-band handoff (also covered by the far sweep)
            -5.5, // the ADR case: -0 versus -5.93e-9
            -2.0, // mid-band handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative far band `[-10.5, -6)`, swept at the
/// conditioning bound: `|z|` runs `12.5` to `47` here, so the envelope
/// runs about `46` to `156` ULP against normal references, plus the
/// subnormal-reference floor past `x = -10`. Covers the sigmoid-arg
/// normality edge and the underflow transition.
fn far_inputs() -> Vec<f32> {
    sweep_inputs(
        -10.5,
        -6.0,
        512,
        &[
            -6.0,  // far-band handoff (also covered by the mid sweep)
            -9.9,  // the sigmoid argument passes -85 here
            -9.95, // WGSL flush floor: the last swept normal-intermediate input
            -10.0, // the sigmoid argument passes -87.3 here (CUDA/HIP only)
            -10.5, // underflow-transition interior (CUDA/HIP only)
            -11.0, // clean zero past here (CUDA/HIP only)
        ],
    )
}

/// Sweep [`GeluTanhOp`](hephaestus_core::GeluTanhOp) on one backend.
///
/// Bounds grade with the `|x|³` conditioning (the argument rounding is
/// dialect-independent; the driver grade stacks on top for WGSL): main
/// `16` ULP on CUDA/HIP and `32` ULP on WGSL, mid `64`/`128` ULP, far
/// `256` ULP on every dialect.
/// The HIP `expf`/`/` rows are the CUDA rows carried unconfirmed — no
/// HIP-published rows were found; any excess on hardware CI is a
/// finding. The mid bound still catches the hunted defect with four
/// orders of margin (`0` against a normal float scores past `1e6` ULP).
/// WGSL drops `x < -9.95`, where `exp(2z)` goes subnormal: below that a
/// flushing driver zeroes the sigmoid while `x` keeps the true result
/// normal — the floor follows the intermediate, not the result. CUDA
/// and HIP sweep the tail as IEEE.
pub fn assert_gelu_tanh_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluTanhOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (16.0, 64.0, 256.0),
        SweepDialect::Wgsl => (32.0, 128.0, 256.0),
    };
    let main = main_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -9.95),
            above_flush_floor(&without_subnormals(&mid), -9.95),
            above_flush_floor(&without_subnormals(&far), -9.95),
        )
    } else {
        (main, mid, far)
    };
    assert_unary_sweep_within::<D, E, GeluTanhOp>(device, ops, "gelu-tanh", &main, max_ulp);
    assert_unary_sweep_within::<D, E, GeluTanhOp>(device, ops, "gelu-tanh-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, GeluTanhOp>(device, ops, "gelu-tanh-far", &far, far_ulp);
}
