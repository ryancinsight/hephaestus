//! Accuracy sweep for [`GeluGradOp`](hephaestus_core::GeluGradOp).
//!
//! The renderings take the input (ADR 0061 Decision 7) and apply the same
//! `erfc` identity as [`GeluOp`](hephaestus_core::GeluOp) to the
//! error-function term: `0.5·erfc(-x/√2) + x·exp(-x²/2)/√(2π)`, with the
//! Gaussian term untouched (its decay is the true asymptotic, not a
//! cancellation artifact). Both infinities guard explicitly (`+∞` to `1`,
//! `-∞` to `0`); the unguarded form makes `±∞·0` in the Gaussian term. The
//! sweep covers the old-form tail defect, the deep negative tail, the
//! structural specials, and — absolutely — the root window near
//! `x = -0.7525`, where relative ULP scoring is vacuous.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, GeluGradOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_abs_within, assert_unary_sweep_within,
    sweep_inputs, without_subnormals,
};

/// 2048 points over each of `[-3, -1.4]` and `[-0.1, 4]`, 256 over the
/// positive tail, plus the band handoffs and the structural specials:
/// everything outside the root window at or above the graded tail bands,
/// swept at the form bound. The positive tail sweeps here because its
/// error vanishes (the result nears `1`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -3.0,
        -1.4,
        2048,
        &[
            -3.0, // mid-band handoff (also covered by the mid sweep)
            -1.4, // root-window handoff (also covered by the root sweep)
        ],
    );
    inputs.extend(sweep_inputs(-0.1, 4.0, 2048, &[-0.1]));
    inputs.extend(sweep_inputs(4.0, 20.0, 256, &[]));
    inputs
}

/// 512 points over the root window `[-1.4, -0.1]`, swept at the absolute
/// floor: the derivative crosses zero near `x = -0.7525`, and near that
/// root any fixed absolute form error scores arbitrarily many relative
/// ULP, so relative scoring would fail a correct form there.
fn root_inputs() -> Vec<f32> {
    sweep_inputs(
        -1.4,
        -0.1,
        512,
        &[
            -1.4,    // root-window handoff (also covered by the main sweep)
            -0.7525, // the root
            -0.1,    // root-window handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the mid band `[-7, -3)`, swept at the conditioning
/// bound (see `gelu.rs`): the `x·inv√2` product rounds, and `erfc`
/// amplifies relative argument error by `2z²`.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -7.0,
        -3.0,
        512,
        &[
            -3.0, // mid-band handoff (also covered by the main sweep)
            -5.5, // the old `1 + erf` term rounds to 0 past here
            -7.0, // far-band handoff (also covered by the far sweep)
        ],
    )
}

/// 512 points over the far band `[-15, -7)`, swept at the conditioning
/// bound at its deepest `z`: the same `2z²` amplification as the mid band.
fn far_inputs() -> Vec<f32> {
    sweep_inputs(
        -15.0,
        -7.0,
        512,
        &[
            -7.0,  // far-band handoff (also covered by the mid sweep)
            -10.0, // deep tail interior
            -12.5, // WGSL flush floor: the last swept subnormal-free input
            -13.0, // past the floor (CUDA/HIP only)
        ],
    )
}

/// Sweep [`GeluGradOp`](hephaestus_core::GeluGradOp) on one backend.
///
/// Bounds grade with the `2z²` conditioning like `gelu.rs`: main `16`
/// ULP, mid `64` ULP on the C dialects and `128` ULP on WGSL, far `256`
/// ULP. The HIP `erfcf` row is the CUDA row carried unconfirmed — no
/// HIP-published row was found; any excess on hardware CI is a finding.
/// The root window sweeps separately at the absolute floor: CUDA `2e-7`,
/// HIP `2e-7`, WGSL `3e-7` (the forms' absolute rounding over unit-scale
/// terms, plus margin). WGSL additionally drops `x < -12.5`, where the
/// true result is subnormal and flushing is permitted; CUDA and HIP sweep
/// the tail as IEEE.
pub fn assert_gelu_grad_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluGradOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp, abs_floor) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (16.0, 64.0, 256.0, 2e-7),
        SweepDialect::Wgsl => (16.0, 128.0, 256.0, 3e-7),
    };
    let main = main_inputs();
    let root = root_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, root, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -12.5),
            above_flush_floor(&without_subnormals(&root), -12.5),
            above_flush_floor(&without_subnormals(&mid), -12.5),
            above_flush_floor(&without_subnormals(&far), -12.5),
        )
    } else {
        (main, root, mid, far)
    };
    assert_unary_sweep_within::<D, E, GeluGradOp>(device, ops, "gelu-grad", &main, max_ulp);
    assert_unary_sweep_abs_within::<D, E, GeluGradOp>(
        device,
        ops,
        "gelu-grad-root",
        &root,
        abs_floor,
    );
    assert_unary_sweep_within::<D, E, GeluGradOp>(device, ops, "gelu-grad-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, GeluGradOp>(device, ops, "gelu-grad-far", &far, far_ulp);
}
