//! Accuracy sweep for [`GeluTanhGradOp`](hephaestus_core::GeluTanhGradOp).
//!
//! The renderings take the input (ADR 0061 Decision 7) and rewrite the
//! textbook `0.5·(1+tanh(z)) + 0.5·x·(1-tanh(z)²)·z'(x)` around the uniform
//! stabilized sigmoid `s = sigmoid(2z)`: `s + 2·x·s·(1-s)·c0·(1+3·c1·x²)`
//! with `1 - s` direct below `w = 2.47` and independent (`sigmoid(-w)`)
//! above (`GELU_TANH_GRAD_CROSSOVER`, the measured optimum — the direct
//! branch keeps the single-form multiplication order, the independent
//! branch pre-groups the saturation). Either saturation at exactly zero
//! returns `s`: past `|x| ≈ 5e19` the `x²` factor overflows, and forming
//! the product would make `0·∞`. Both infinities guard explicitly (`+∞`
//! to `1`, `-∞` to `0`); the unguarded form makes `±∞·0`. The sweep
//! covers the crossover, the cubic-driven tail, the structural specials,
//! and — absolutely — the root window near `x = -0.7525`, where relative
//! ULP scoring is vacuous. The negative side grades by `|x|³`
//! conditioning, as `gelu_tanh.rs` does.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, GeluTanhGradOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_abs_within, assert_unary_sweep_within,
    sweep_inputs, without_subnormals,
};

/// 512 points over `[-2, -1.35]`, 2048 over `[-0.15, 4]`, 256 over the
/// positive tail, plus the band handoffs, the crossover, and the
/// structural specials: everything outside the root window at or above
/// the conditioning mid band, swept at the form bound. The positive tail
/// sweeps here because its error vanishes (the result nears `1`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -2.0,
        -1.35,
        512,
        &[
            -2.0,  // mid-band handoff (also covered by the mid sweep)
            -1.35, // root-window handoff (also covered by the root sweep)
        ],
    );
    inputs.extend(sweep_inputs(
        -0.15,
        4.0,
        2048,
        &[
            -0.15, // root-window handoff (also covered by the root sweep)
            0.0,   // +0 (the builder adds -0)
            1.42,  // the crossover sits at w = 2.47 here (both forms beside it)
        ],
    ));
    inputs.extend(sweep_inputs(4.0, 20.0, 256, &[]));
    inputs
}

/// 512 points over the root window `[-1.35, -0.15]`, swept at the absolute
/// floor: the derivative crosses zero near `x = -0.7525`, and near that
/// root any fixed absolute form error scores arbitrarily many relative
/// ULP, so relative scoring would fail a correct form there.
fn root_inputs() -> Vec<f32> {
    sweep_inputs(
        -1.35,
        -0.15,
        512,
        &[
            -1.35,   // root-window handoff (also covered by the main sweep)
            -0.7525, // the root
            -0.15,   // root-window handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative mid band `[-6, -2)`, swept at the
/// conditioning bound: `|z|` runs `1.9` to `12.5` here, so the envelope
/// runs about `16` to `50` ULP (the forward envelope plus the longer
/// product chain).
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -6.0,
        -2.0,
        512,
        &[
            -6.0, // far-band handoff (also covered by the far sweep)
            -5.5, // the forward form's zero crossing (shared shape)
            -2.0, // mid-band handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative far band `[-10.5, -6)`, swept at the
/// conditioning bound: `|z|` runs `12.5` to `47` here, so the envelope
/// runs about `50` to `160` ULP against normal references, plus the
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

/// Sweep [`GeluTanhGradOp`](hephaestus_core::GeluTanhGradOp) on one backend.
///
/// Bounds grade with the `|x|³` conditioning (the argument rounding is
/// dialect-independent; the driver grade stacks on top for WGSL): main
/// `16` ULP on CUDA/HIP and `32` ULP on WGSL, mid `64`/`128` ULP, far
/// `256` ULP on every dialect. The HIP `expf`/`/` rows are the CUDA rows
/// carried unconfirmed — no HIP-published rows were found; any excess on
/// hardware CI is a finding. The root window sweeps separately at the
/// absolute floor: CUDA `2e-7`, HIP `2e-7`, WGSL `3e-7` (the forms'
/// absolute rounding over unit-scale terms, plus margin). WGSL drops
/// `x < -9.95`, where `exp(2z)` goes subnormal — the floor follows the
/// intermediate, not the result. CUDA and HIP sweep the tail as IEEE.
pub fn assert_gelu_tanh_grad_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluTanhGradOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp, abs_floor) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (16.0, 64.0, 256.0, 2e-7),
        SweepDialect::Wgsl => (32.0, 128.0, 256.0, 3e-7),
    };
    let main = main_inputs();
    let root = root_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, root, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -9.95),
            above_flush_floor(&without_subnormals(&root), -9.95),
            above_flush_floor(&without_subnormals(&mid), -9.95),
            above_flush_floor(&without_subnormals(&far), -9.95),
        )
    } else {
        (main, root, mid, far)
    };
    assert_unary_sweep_within::<D, E, GeluTanhGradOp>(
        device,
        ops,
        "gelu-tanh-grad",
        &main,
        max_ulp,
    );
    assert_unary_sweep_abs_within::<D, E, GeluTanhGradOp>(
        device,
        ops,
        "gelu-tanh-grad-root",
        &root,
        abs_floor,
    );
    assert_unary_sweep_within::<D, E, GeluTanhGradOp>(
        device,
        ops,
        "gelu-tanh-grad-mid",
        &mid,
        mid_ulp,
    );
    assert_unary_sweep_within::<D, E, GeluTanhGradOp>(
        device,
        ops,
        "gelu-tanh-grad-far",
        &far,
        far_ulp,
    );
}
