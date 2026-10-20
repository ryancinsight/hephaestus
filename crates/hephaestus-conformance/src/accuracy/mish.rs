//! Accuracy sweep for [`MishOp`](hephaestus_core::MishOp).
//!
//! The renderings evaluate `x·tanh(softplus(x))` around the stabilized
//! two-sided softplus (the same halves slice 2 pins for
//! [`SoftplusOp`](hephaestus_core::SoftplusOp), shared macros): the naive
//! `log(1+exp(x))` rounds the softplus to `0` below `x = -16.6`, zeroing
//! the whole product where the value stays a normal float down to
//! `x ≈ -91.5`, and overflows `exp` past `x = 88.73`. `x = -∞` guards to
//! `-0` (the unguarded form makes `-∞·0`). The sweep covers the softplus
//! knees, the `exp` overflow knees, the deep negative tail, and the
//! structural specials. The negative tail splits at `exp` normality
//! (`x = -87`), as `silu.rs` does: `tanh` is near-linear over the tail,
//! so the same subnormal-intermediate envelope applies.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, MishOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// 4096 points over `[-10, 10]`, 256 over the positive tail, plus the
/// tail-band handoff and the structural specials: everything at or above
/// the negative mid band, swept at the form bound. The positive tail
/// sweeps here because its error vanishes (the result nears `x`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -10.0,
        10.0,
        4096,
        &[
            88.73, 89.0, 100.0, // positive saturation interior
            -10.0, // tail-band handoff (also covered by the mid sweep)
            0.0,   // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(10.0, 20.0, 256, &[]));
    inputs
}

/// 512 points over the negative mid band `[-87, -10)`, swept at the
/// driver-`exp` bound: the rendering is transparent `x·tanh(exp(x))`
/// with `tanh` near-linear there, so the error is the driver's `exp`
/// grade at large arguments — the same loose grade the `sigmoid` tail
/// band records.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -87.0,
        -10.0,
        512,
        &[
            -87.0, // far-band handoff (also covered by the far sweep)
            -16.6, // the naive softplus rounds to 0 below here
            -20.0, // the ADR case: 0 versus -4.12e-8
            -10.0, // tail-band handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative far band `[-93, -87)`, swept at the
/// subnormal-intermediate envelope of `silu.rs`: `exp(x)` goes subnormal
/// while `x` keeps the result normal (`tanh` stays near-linear, so the
/// `tanh` of the softplus preserves the intermediate's error 1:1).
/// Covers the `exp` knees and the deep underflow lanes.
fn far_inputs() -> Vec<f32> {
    sweep_inputs(
        -93.0,
        -87.0,
        512,
        &[
            -87.0,  // far-band handoff (also covered by the mid sweep)
            -88.72, // just above the exp knee
            -88.73, // just past it: exp overflows under the old form
            -89.0,  // saturation interior
            -91.5,  // underflow-transition interior
            -92.0,  // subnormal-result interior (CUDA/HIP only)
            -110.0, // deep underflow lane (CUDA/HIP only)
            -120.0, // clean zero past here (CUDA/HIP only)
        ],
    )
}

/// Sweep [`MishOp`](hephaestus_core::MishOp) on one backend.
///
/// Bounds: main CUDA `8` ULP (the stabilized softplus at `log1pf` `1`
/// plus `expf` `2`, `tanhf` at `2`, the product, plus margin); HIP `8`
/// ULP (the CUDA rows carried unconfirmed — any excess on hardware CI is
/// a finding); WGSL `16` ULP (the two-band softplus at its `softplus.rs`
/// main analysis; `tanh` only above `sp = 0.036`, past the bad zone
/// four sweep rounds pin at `x = -3.40` on both CI drivers (the naive
/// quotient-form `tanh` carries constant absolute error, hence the
/// `1/s` profile), and the corrected `x·S·(1-S²/3)` composition below,
/// whose Taylor remainder peaks at `3.9` ULP at the `-3.3` edge; the
/// product, plus margin). The mid band carries the driver-`exp` grade:
/// CUDA/HIP `8` ULP, WGSL `64` ULP (as `silu.rs`). The far band carries
/// the subnormal-intermediate envelope (`≈ 230` at the deepest
/// normal-result input, as `silu.rs` derives): CUDA/HIP `256`, WGSL
/// `256` over the surviving handoff/specials. WGSL drops `x < -87.0`
/// (matching `sigmoid.rs`): below `exp` normality a flushing driver
/// zeroes the softplus intermediate while `x` keeps the true result
/// normal — the floor follows the intermediate, not the result. CUDA
/// and HIP sweep the tail as IEEE.
pub fn assert_mish_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    MishOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (8.0, 8.0, 256.0),
        SweepDialect::Wgsl => (16.0, 64.0, 256.0),
    };
    let main = main_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -87.0),
            above_flush_floor(&without_subnormals(&mid), -87.0),
            above_flush_floor(&without_subnormals(&far), -87.0),
        )
    } else {
        (main, mid, far)
    };
    assert_unary_sweep_within::<D, E, MishOp>(device, ops, "mish", &main, max_ulp);
    assert_unary_sweep_within::<D, E, MishOp>(device, ops, "mish-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, MishOp>(device, ops, "mish-far", &far, far_ulp);
}
