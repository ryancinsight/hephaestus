//! Accuracy sweep for [`ErfcOp`](hephaestus_core::ErfcOp).
//!
//! The renderings evaluate the complementary error function without the
//! `1 - erf(x)` cancellation: the C dialects call native `erfc`, while
//! WGSL spells the Numerical Recipes `erfcc` Chebyshev form
//! (`t·exp(-z²-1.26551223+t·P(t))`, `t = 1/(1+z/2)`) with the `2 - ans`
//! branch for negative `x`. The sweep covers the old-form zero crossing
//! near `x = 2.75`, the deep tail where the value stays a normal float
//! down to `x ≈ 9`, and the structural specials.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, ErfcOp, UnaryExpr};

use super::runner::with_normal_erfc_result;
use super::{SweepDialect, assert_unary_sweep_within, sweep_inputs, without_subnormals};

/// 4096 points over `[-4, 3]`, 256 over the negative tail, plus the
/// old-form zero crossing, the tail-band handoff, and the structural
/// specials: everything at or below the driver-`exp` tail band, swept at
/// the form bound. The negative side sweeps here because its error
/// vanishes (the result nears `2`, so the `exp` perturbation is sub-ULP).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -4.0,
        3.0,
        4096,
        &[
            2.75, // the old `1 - erf` form rounds to 0 past here
            3.0,  // tail-band handoff (also covered by the tail sweep)
            0.0,  // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(-12.0, -4.0, 256, &[]));
    inputs
}

/// 256 points over the mid band `(3, 6]`, swept below the far envelope:
/// the `z²` rounding (absolute `≈ 2.4e-7` there) and the driver's `exp`
/// grade at arguments past `-10` both grow with `z`, so each band carries
/// the bound at its deepest `z`.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        3.0,
        6.0,
        256,
        &[
            3.0, // mid-band handoff (also covered by the main sweep)
            6.0, // far-band handoff (also covered by the far sweep)
        ],
    )
}

/// 512 points over the far band `(6, 10.5]`, swept at the envelope bound:
/// the rendering is transparent `exp` of a large negative argument times
/// a unit-scale polynomial there, so the error is the `z²` rounding
/// (absolute, amplified 1:1 by `exp`) plus the driver's `exp` grade at
/// large arguments — the same loose grade the `expm1` knee and the
/// `sigmoid` tail bands record.
fn far_inputs() -> Vec<f32> {
    sweep_inputs(
        6.0,
        10.5,
        512,
        &[
            6.0, // far-band handoff (also covered by the mid sweep)
            8.3, // approaching subnormal results
            8.9, // former fixed cap; the reference remains normal beyond it
            9.0, 9.19, 9.194_549, 10.0, // deep tail interior (CUDA/HIP only)
        ],
    )
}

fn wgsl_far_inputs(inputs: &[f32]) -> Vec<f32> {
    with_normal_erfc_result(&without_subnormals(inputs), |x| x)
}

/// Sweep [`ErfcOp`](hephaestus_core::ErfcOp) on one backend.
///
/// Existing acceptance targets remain unchanged: CUDA `5` ULP, HIP `4` ULP,
/// WGSL main `32`, mid `128`, and far `256` ULP. The HIP row remains
/// unconfirmed on hardware. These sampled targets do not establish a
/// portable composition bound from WGSL intrinsic allowances: an allowed
/// `exp` perturbation can exceed the mid target (tracked in the backlog).
/// WGSL retains every far input whose f64 `erfc` reference is f32-normal,
/// including `9` and `9.19`; it excludes subnormal reference results because
/// the dialect permits flushing. CUDA and HIP retain their IEEE tail sweep.
pub fn assert_erfc_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    ErfcOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp) = match dialect {
        SweepDialect::CudaC => (5.0, 5.0, 5.0),
        SweepDialect::HipC => (4.0, 4.0, 4.0),
        SweepDialect::Wgsl => (32.0, 128.0, 256.0),
    };
    let main = main_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals and caps the subnormal-result tail;
    // the other dialects sweep everything as IEEE.
    let (main, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            without_subnormals(&main),
            without_subnormals(&mid),
            wgsl_far_inputs(&far),
        )
    } else {
        (main, mid, far)
    };
    assert_unary_sweep_within::<D, E, ErfcOp>(device, ops, "erfc", &main, max_ulp);
    assert_unary_sweep_within::<D, E, ErfcOp>(device, ops, "erfc-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, ErfcOp>(device, ops, "erfc-far", &far, far_ulp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hephaestus_core::UnaryValue;

    #[test]
    fn wgsl_sweep_keeps_directed_normal_erfc_tails() {
        let inputs = wgsl_far_inputs(&far_inputs());
        for x in [9.0, 9.19, 9.194_549] {
            assert!(ErfcOp::apply(f64::from(x)) >= f64::from(f32::MIN_POSITIVE));
            assert!(inputs.contains(&x), "normal erfc tail removed at {x}");
        }
        let last_normal = 9.194_549_f32;
        let boundary = wgsl_far_inputs(&[last_normal, last_normal.next_up()]);
        assert_eq!(boundary, [last_normal]);
        assert!(inputs.iter().any(|x| x.is_nan()));
        assert!(inputs.contains(&f32::INFINITY));
        assert!(inputs.contains(&f32::NEG_INFINITY));
    }
}
