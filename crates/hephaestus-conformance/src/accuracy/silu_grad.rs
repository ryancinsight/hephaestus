//! Accuracy sweep for [`SiluGradOp`](hephaestus_core::SiluGradOp).
//!
//! The renderings evaluate `s·(1+x·(1-s))` with the stabilized two-sided
//! sigmoid for `s`, taking the input (ADR 0061 Decision 7): `1 - s` goes
//! direct below `x = 2.63` and independent (`sigmoid(-x)`) above
//! (`SILU_GRAD_CROSSOVER`, the measured optimum — uniform-direct
//! amplifies the sigmoid's own error through the subtraction past the
//! crossover). Both infinities guard explicitly (`+∞` to `1`, `-∞` to
//! `0`); the unguarded form makes `±∞·0`. The sweep covers the `exp`
//! knees, the crossover, the deep negative tail, the structural specials,
//! and — absolutely — the root window near `x = -1.28`, where relative
//! ULP scoring is vacuous. The negative tail splits at `exp` normality
//! (`x = -87`), as `silu.rs` does.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, SiluGradOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_abs_within, assert_unary_sweep_within,
    sweep_inputs, without_subnormals,
};

/// 2048 points over each of `[-10, -1.8]` and `[-0.8, 10]`, 256 over the
/// positive tail, plus the band handoffs, the crossover, and the
/// structural specials: everything outside the root window at or above
/// the negative mid band, swept at the form bound. The positive tail
/// sweeps here because its error vanishes (the result nears `1`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -10.0,
        -1.8,
        2048,
        &[
            -10.0, // tail-band handoff (also covered by the mid sweep)
            -1.8,  // root-window handoff (also covered by the root sweep)
        ],
    );
    inputs.extend(sweep_inputs(
        -0.8,
        10.0,
        2048,
        &[
            -0.8, // root-window handoff (also covered by the root sweep)
            0.0,  // +0 (the builder adds -0)
            2.63, // the crossover (both branches evaluate beside it)
        ],
    ));
    inputs.extend(sweep_inputs(10.0, 120.0, 256, &[]));
    inputs
}

/// 512 points over the root window `[-1.8, -0.8]`, swept at the absolute
/// floor: the derivative crosses zero near `x = -1.28`, and near that root
/// any fixed absolute form error scores arbitrarily many relative ULP, so
/// relative scoring would fail a correct form there.
fn root_inputs() -> Vec<f32> {
    sweep_inputs(
        -1.8,
        -0.8,
        512,
        &[
            -1.8,  // root-window handoff (also covered by the main sweep)
            -1.28, // the root
            -0.8,  // root-window handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative mid band `[-87, -10)`, swept at the
/// driver-`exp` bound: `exp(x)` stays normal, so the error is the builtin
/// grade transferred 1:1 relatively — the same grade the `sigmoid` tail
/// band records.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -87.0,
        -10.0,
        512,
        &[
            -87.0, // far-band handoff (also covered by the far sweep)
            -20.0, // saturation interior
            -10.0, // tail-band handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative far band `[-93, -87)`, swept at the
/// subnormal-intermediate envelope of `silu.rs`: `exp(x)` goes subnormal
/// while `1 + x` keeps the result normal, so the intermediate's fixed
/// `2^-149` rounding amplifies by `|1+x|` against the result's ULP.
/// Covers the ADR defect window, the `exp` knees, and the deep underflow
/// lanes.
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
            -91.86, // the ADR defect-window bottom (CUDA/HIP only)
            -92.0,  // subnormal-result interior (CUDA/HIP only)
            -110.0, // deep underflow lane (CUDA/HIP only)
            -120.0, // clean zero past here (CUDA/HIP only)
        ],
    )
}

/// Sweep [`SiluGradOp`](hephaestus_core::SiluGradOp) on one backend.
///
/// Bounds: main CUDA `8` ULP (two stabilized sigmoids at `expf` `2` plus
/// `/` `2` each, products and the sum, plus margin); HIP `8` ULP (the
/// CUDA rows carried unconfirmed — any excess on hardware CI is a
/// finding); WGSL `12` ULP (the second sigmoid's error enters through
/// `x·σ(−x) ≤ 0.19` past the crossover, so the two driver grades never
/// stack fully). The root window sweeps separately at the absolute floor:
/// CUDA `2e-7`, HIP `2e-7`, WGSL `3e-7` (the forms' absolute rounding
/// over unit-scale terms, plus margin). The mid band carries the
/// driver-`exp` grade: CUDA/HIP `8` ULP, WGSL `64` ULP (as `silu.rs`).
/// The far band carries the subnormal-intermediate envelope: at
/// `x = -91.86` the worst case is `90.86·2.5·2^-149/2^-149 ≈ 227`
/// against the bottom-normal-bin ULP, so CUDA/HIP carry `256`; WGSL
/// carries `256` over the surviving handoff/specials. WGSL drops
/// `x < -87.0` (matching `sigmoid.rs`): below `exp` normality a flushing
/// driver zeroes the intermediate while `1 + x` keeps the true result
/// normal — the floor follows the intermediate, not the result. CUDA and
/// HIP sweep the tail as IEEE.
pub fn assert_silu_grad_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SiluGradOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp, abs_floor) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (8.0, 8.0, 256.0, 2e-7),
        SweepDialect::Wgsl => (12.0, 64.0, 256.0, 3e-7),
    };
    let main = main_inputs();
    let root = root_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, root, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -87.0),
            above_flush_floor(&without_subnormals(&root), -87.0),
            above_flush_floor(&without_subnormals(&mid), -87.0),
            above_flush_floor(&without_subnormals(&far), -87.0),
        )
    } else {
        (main, root, mid, far)
    };
    assert_unary_sweep_within::<D, E, SiluGradOp>(device, ops, "silu-grad", &main, max_ulp);
    assert_unary_sweep_abs_within::<D, E, SiluGradOp>(
        device,
        ops,
        "silu-grad-root",
        &root,
        abs_floor,
    );
    assert_unary_sweep_within::<D, E, SiluGradOp>(device, ops, "silu-grad-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, SiluGradOp>(device, ops, "silu-grad-far", &far, far_ulp);
}
