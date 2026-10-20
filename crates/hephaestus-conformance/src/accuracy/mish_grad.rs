//! Accuracy sweep for [`MishGradOp`](hephaestus_core::MishGradOp).
//!
//! The renderings evaluate `t + x·(1-t²)·s` with `t = tanh(sp)`, the
//! stabilized two-sided softplus for `sp`, and the stabilized two-sided
//! sigmoid for `s`, taking the input (ADR 0061 Decision 7): `1 - t²` goes
//! direct below `sp = 1.55` and `sech²` (`4u/(1+u)²`, `u = exp(-2·sp)`)
//! above (`MISH_GRAD_CROSSOVER`, compared on `sp`, not `t` —
//! uniform-direct amplifies the `tanh` input error through the subtraction
//! past the crossover). Both infinities guard explicitly (`+∞` to `1`,
//! `-∞` to `0`); the unguarded form makes `±∞·0`. The sweep covers the
//! softplus knees, the crossover, the deep negative tail, the structural
//! specials, and — absolutely — the root window near `x = -1.2`, where
//! relative ULP scoring is vacuous. The negative tail splits at `exp`
//! normality (`x = -87`), as `silu.rs` does.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, MishGradOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_abs_within, assert_unary_sweep_within,
    sweep_inputs, without_subnormals,
};

/// 2048 points over each of `[-10, -1.7]` and `[-0.7, 10]`, 256 over the
/// positive tail, plus the band handoffs, the crossover, and the
/// structural specials: everything outside the root window at or above
/// the negative mid band, swept at the form bound. The positive tail
/// sweeps here because its error vanishes (the result nears `1`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -10.0,
        -1.7,
        2048,
        &[
            -10.0, // tail-band handoff (also covered by the mid sweep)
            -1.7,  // root-window handoff (also covered by the root sweep)
        ],
    );
    inputs.extend(sweep_inputs(
        -0.7,
        10.0,
        2048,
        &[
            -0.7, // root-window handoff (also covered by the root sweep)
            0.0,  // +0 (the builder adds -0)
            1.32, // the crossover sits at sp = 1.55 here (both forms beside it)
        ],
    ));
    inputs.extend(sweep_inputs(10.0, 20.0, 256, &[]));
    inputs
}

/// 512 points over the root window `[-1.7, -0.7]`, swept at the absolute
/// floor: the derivative crosses zero near `x = -1.2`, and near that root
/// any fixed absolute form error scores arbitrarily many relative ULP, so
/// relative scoring would fail a correct form there.
fn root_inputs() -> Vec<f32> {
    sweep_inputs(
        -1.7,
        -0.7,
        512,
        &[
            -1.7, // root-window handoff (also covered by the main sweep)
            -1.2, // the root
            -0.7, // root-window handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative mid band `[-87, -10)`, swept at the
/// driver-`exp` bound: the rendering is transparent `tanh`/`sigmoid` of
/// the softplus tail there, so the error is the driver's `exp` grade at
/// large arguments — the same loose grade the `sigmoid` tail band records.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -87.0,
        -10.0,
        512,
        &[
            -87.0, // far-band handoff (also covered by the far sweep)
            -16.6, // the naive softplus rounds to 0 below here
            -20.0, // the ADR case: 5% off there
            -10.0, // tail-band handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative far band `[-93, -87)`, swept at the
/// subnormal-intermediate envelope of `silu.rs`: `exp(x)` goes subnormal
/// while `x` keeps the result normal, so the intermediate's fixed
/// `2^-149` rounding amplifies by `|x|` against the result's ULP.
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

/// Sweep [`MishGradOp`](hephaestus_core::MishGradOp) on one backend.
///
/// Bounds: main CUDA `8` ULP (the `tanh` term at softplus `4` plus
/// `tanhf` `2`, the `sech²` term at softplus plus `expf` plus `/`, the
/// sigmoid, the sum, plus margin); HIP `8` ULP (the CUDA rows carried
/// unconfirmed — any excess on hardware CI is a finding); WGSL `12` ULP
/// (the `tanh` term at two-band softplus plus driver grade above
/// `sp = 0.018` — past the bad-`tanh` edge two sweep rounds pin at
/// `x = -4.2` on Metal — the corrected `S·(1-S²/3)` composition below
/// (Taylor remainder `0.24` ULP at the `-4` edge), the `sech²` and
/// sigmoid terms, the sum, plus margin). The root
/// window sweeps separately at the absolute floor: CUDA `2e-7`, HIP
/// `2e-7`, WGSL `3e-7` (the forms' absolute rounding over unit-scale
/// terms, plus margin). The mid band carries the driver-`exp` grade:
/// CUDA/HIP `8` ULP, WGSL `64` ULP (as `silu.rs`). The far band carries
/// the subnormal-intermediate envelope (`≈ 230` at the deepest
/// normal-result input, as `silu.rs` derives): CUDA/HIP `256`, WGSL
/// `256` over the surviving handoff/specials. WGSL drops `x < -87.0`
/// (matching `sigmoid.rs`): below `exp` normality a flushing driver
/// zeroes the softplus intermediate while `x` keeps the true result
/// normal — the floor follows the intermediate, not the result. CUDA
/// and HIP sweep the tail as IEEE.
pub fn assert_mish_grad_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    MishGradOp: UnaryExpr<E::Dialect>,
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
    assert_unary_sweep_within::<D, E, MishGradOp>(device, ops, "mish-grad", &main, max_ulp);
    assert_unary_sweep_abs_within::<D, E, MishGradOp>(
        device,
        ops,
        "mish-grad-root",
        &root,
        abs_floor,
    );
    assert_unary_sweep_within::<D, E, MishGradOp>(device, ops, "mish-grad-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, MishGradOp>(device, ops, "mish-grad-far", &far, far_ulp);
}
