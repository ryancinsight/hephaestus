//! Accuracy sweep for [`GeluOp`](hephaestus_core::GeluOp).
//!
//! The renderings apply the `1 + erf(z) = erfc(-z)` identity to the
//! textbook `0.5·x·(1 + erf(x/√2))`: the C dialects evaluate
//! `0.5·x·erfc(-x/√2)` around the native builtin, while WGSL evaluates the
//! same identity around the Chebyshev `erfc` form (see `erfc.rs`). Both
//! guard `x = -∞` (which otherwise forms `-∞·0`) to `-0`. The sweep covers
//! the old-form zero crossing near `x = -5.5`, the deep negative tail
//! where the value stays a normal float, and the structural specials. The
//! forward root at `x = 0` needs no absolute window: the form is
//! multiplicative (`0.5·x` times a smooth near-one factor), so it stays
//! relatively accurate through the root.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, GeluOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// 4096 points over `[-3, 4]`, 256 over the positive tail, plus the
/// mid-band handoff and the structural specials: everything at or above
/// the graded tail bands, swept at the form bound. The positive tail
/// sweeps here because its error vanishes (the result nears `x`, so the
/// `erfc` perturbation is sub-ULP).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -3.0,
        4.0,
        4096,
        &[
            -3.0, // mid-band handoff (also covered by the mid sweep)
            0.0,  // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(4.0, 20.0, 256, &[]));
    inputs
}

/// 512 points over the mid band `[-7, -3)`, swept at the conditioning
/// bound: the `x·inv√2` product rounds (≈ 1 ULP with the constant), and
/// `erfc` amplifies relative argument error by `2z²` (log-log slope via
/// the Mills ratio), so the error grows quadratically down the tail and
/// each band carries the bound at its deepest `z`.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -7.0,
        -3.0,
        512,
        &[
            -3.0, // mid-band handoff (also covered by the main sweep)
            -5.5, // the old `1 + erf` form rounds to 0 past here
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
            -12.6, -13.0, // past the floor (CUDA/HIP only)
        ],
    )
}

fn wgsl_inputs(inputs: &[f32]) -> Vec<f32> {
    above_flush_floor(&without_subnormals(inputs), -12.5)
}

/// Sweep [`GeluOp`](hephaestus_core::GeluOp) on one backend.
///
/// Bounds grade with the `2z²` conditioning: main `16` ULP on the C
/// dialects (`2·2.12² + erfcf 4 + products`, plus margin) and `32` ULP
/// on WGSL (the Chebyshev evaluation's driver-`exp` over arguments past
/// `-5`, as `erfc.rs` derives; lavapipe measures `18.4` worst at
/// `x = -2.89`), mid `64` ULP on the C dialects and `128` ULP on WGSL
/// (the WGSL mid adds the driver's
/// large-argument `exp` grade inside the Chebyshev evaluation), far `256`
/// ULP on every dialect. The HIP `erfcf` row is the CUDA row carried
/// unconfirmed — no HIP-published row was found; any excess on hardware CI
/// is a finding. The far bound still catches the hunted defect with four
/// orders of margin (`0` against a normal float scores past `1e6` ULP).
/// WGSL additionally drops `x < -12.5`, where the true result is subnormal
/// and flushing is permitted; CUDA and HIP sweep the tail as IEEE.
pub fn assert_gelu_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluOp: UnaryExpr<E::Dialect>,
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
        (wgsl_inputs(&main), wgsl_inputs(&mid), wgsl_inputs(&far))
    } else {
        (main, mid, far)
    };
    assert_unary_sweep_within::<D, E, GeluOp>(device, ops, "gelu", &main, max_ulp);
    assert_unary_sweep_within::<D, E, GeluOp>(device, ops, "gelu-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, GeluOp>(device, ops, "gelu-far", &far, far_ulp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hephaestus_core::UnaryValue;

    #[test]
    fn wgsl_sweep_keeps_directed_normal_gelu_tails() {
        let inputs = wgsl_inputs(&far_inputs());
        for x in [-12.6, -13.0] {
            assert!(GeluOp::apply(f64::from(x)).abs() >= f64::from(f32::MIN_POSITIVE));
            assert!(inputs.contains(&x), "normal GELU tail removed at {x}");
        }
    }
}
