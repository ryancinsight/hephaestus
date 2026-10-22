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

use super::runner::with_normal_unary_result;

use super::{SweepDialect, assert_unary_sweep_within, sweep_inputs, without_subnormals};

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
            -3.0,                        // mid-band handoff (also covered by the mid sweep)
            f32::from_bits(0xc036_f400), // exact allowed-exp countermodel
            0.0,                         // +0 (the builder adds -0)
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
            -3.0,                        // mid-band handoff (also covered by the main sweep)
            -5.5,                        // the old `1 + erf` form rounds to 0 past here
            -7.0,                        // far-band handoff (also covered by the far sweep)
            f32::from_bits(0xc0df_ffff), // adjacent unscaled-composer handoff
            f32::from_bits(0xc0e0_0001), // adjacent factored-composer handoff
        ],
    )
}

/// 512 points over the far band `[-15, -7)`, swept at the unchanged
/// conditioning bound. WGSL factors the Gaussian throughout this band
/// so that the rounded `x/sqrt(2)` argument enters only the correction.
fn far_inputs() -> Vec<f32> {
    sweep_inputs(
        -15.0,
        -7.0,
        512,
        &[
            -7.0,                        // far-band handoff (also covered by the mid sweep)
            f32::from_bits(0xc0df_ffff), // adjacent unscaled-composer handoff
            f32::from_bits(0xc0e0_0001), // adjacent factored-composer handoff
            f32::from_bits(0xc14f_e3f4), // retained unscaled-exp source countermodels
            f32::from_bits(0xc149_05b3),
            f32::from_bits(0xc144_6f7a),
            f32::from_bits(0xc13e_0c2d),
            -10.0, // deep tail interior
            -12.5, // former fixed floor; the erfc intermediate stays normal below it
            -12.6,
            -13.0,                       // retained normal erfc/intermediate controls
            f32::from_bits(0xc14f_ffff), // retained control above the former -13 handoff
            f32::from_bits(0xc150_0001), // retained control below the former -13 handoff
            f32::from_bits(0xc150_014b), // old gradual model exceeds 256 ULP
            f32::from_bits(0xc150_0c85), // first old-selector exclusion
            f32::from_bits(0xc151_818a), // old gradual-model worst tail witness
            -13.1,                       // normal final result whose erfc intermediate is subnormal
            f32::from_bits(0xc152_5705), // penultimate normal-result input
            f32::from_bits(0xc152_5706), // last normal-result input
            f32::from_bits(0xc152_5707), // adjacent subnormal-result control
        ],
    )
}

fn wgsl_inputs(inputs: &[f32]) -> Vec<f32> {
    with_normal_unary_result::<GeluOp>(&without_subnormals(inputs))
}

/// Sweep [`GeluOp`](hephaestus_core::GeluOp) on one backend.
///
/// Existing targets remain unchanged: CUDA/HIP main `16`, mid `64`, far
/// `256` ULP; WGSL main `32`, mid `128`, far `256` ULP. The HIP row remains
/// unconfirmed on hardware. A legal WGSL intrinsic perturbation can exceed
/// the main target; these sampled targets have no portable composition proof.
/// WGSL tail selection follows final reference normality, including the
/// last normal-result input. Below the existing far-band boundary `-7`,
/// the shared Chebyshev correction factors out the Gaussian, avoiding
/// quadratic amplification of the rounded erfc argument and subnormal
/// erfc intermediates in the deepest normal-result tail.
/// Source/reference checks do not establish device or portable accuracy.
/// CUDA and HIP retain their IEEE tail sweep.
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
        for x in [
            -12.6,
            -13.0,
            -13.1,
            f32::from_bits(0xc152_5705),
            f32::from_bits(0xc152_5706),
        ] {
            assert!(GeluOp::apply(f64::from(x)).abs() >= f64::from(f32::MIN_POSITIVE));
            assert!(inputs.contains(&x), "normal GELU tail removed at {x}");
        }
        let next = f32::from_bits(0xc152_5707);
        assert!(GeluOp::apply(f64::from(next)).abs() < f64::from(f32::MIN_POSITIVE));
        assert!(!inputs.contains(&next));
    }

    #[test]
    fn wgsl_selection_preserves_existing_gelu_coverage_and_specials() {
        for inputs in [main_inputs(), mid_inputs(), far_inputs()] {
            let selected = wgsl_inputs(&inputs);
            let previous =
                super::super::runner::with_normal_erfc_result(&without_subnormals(&inputs), |x| {
                    -x * core::f32::consts::FRAC_1_SQRT_2
                });
            for x in previous {
                assert!(selected.iter().any(|y| y.to_bits() == x.to_bits()));
            }
        }
    }
    #[test]
    fn wgsl_far_sweep_retains_unscaled_exponent_regressions_and_handoff() {
        let _trace = super::super::runner::trace();
        let _test = tracing::info_span!(
            "gelu_far_regression",
            contract = "unscaled_exponent_and_handoff_selection"
        )
        .entered();
        let inputs = {
            let _arrange = tracing::info_span!("arrange").entered();
            far_inputs()
        };
        let selected = {
            let _act = tracing::info_span!("act").entered();
            wgsl_inputs(&inputs)
        };
        let _assert = tracing::info_span!("assert").entered();
        for bits in [
            0xc14f_e3f4,
            0xc149_05b3,
            0xc144_6f7a,
            0xc13e_0c2d,
            0xc0df_ffff,
            0xc0e0_0000,
            0xc0e0_0001,
        ] {
            let retained = selected.iter().any(|x| x.to_bits() == bits);
            tracing::info!(
                input_bits = bits,
                retained,
                "normal GELU far regression selection"
            );
            assert!(retained, "normal GELU far regression removed: {bits:08x}");
        }
    }

    #[test]
    fn wgsl_sweep_retains_intrinsic_countermodel() {
        let _trace = super::super::runner::trace();
        let _assert = tracing::info_span!(
            "assert",
            op = "gelu",
            contract = "exact_countermodel_selection"
        )
        .entered();
        let witness = f32::from_bits(0xc036_f400);
        let inputs = wgsl_inputs(&main_inputs());
        assert!(
            inputs.contains(&witness),
            "exact intrinsic countermodel input removed: {:08x}",
            witness.to_bits()
        );
    }
}
