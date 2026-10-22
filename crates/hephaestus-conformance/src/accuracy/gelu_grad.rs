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

use super::runner::with_normal_unary_result;

use super::{
    SweepDialect, assert_unary_sweep_abs_within, assert_unary_sweep_within, sweep_inputs,
    without_subnormals,
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
            -12.5, // former fixed floor; the erfc intermediate stays normal below it
            -12.6,
            -13.0,                       // retained normal erfc/intermediate controls
            f32::from_bits(0xc14f_ffff), // adjacent original-composer handoff
            f32::from_bits(0xc150_0001), // adjacent scaled-composer handoff
            f32::from_bits(0xc150_0c85), // first old-selector exclusion
            -13.1,                       // normal final result whose erfc intermediate is subnormal
            -13.3,                       // normal final result whose Gaussian is also subnormal
            f32::from_bits(0xc153_765b), // Gaussian first becomes subnormal
            f32::from_bits(0xc155_7858), // penultimate normal-result input
            f32::from_bits(0xc155_7859), // last normal-result input
            f32::from_bits(0xc155_785a), // adjacent subnormal-result control
        ],
    )
}

fn wgsl_inputs(inputs: &[f32]) -> Vec<f32> {
    with_normal_unary_result::<GeluGradOp>(&without_subnormals(inputs))
}

/// Sweep [`GeluGradOp`](hephaestus_core::GeluGradOp) on one backend.
///
/// Existing targets remain unchanged: all main bands `16` ULP; CUDA/HIP
/// mid `64`, WGSL mid `128`, all far bands `256` ULP. The root window keeps
/// its absolute floor: CUDA/HIP `2e-7`, WGSL `3e-7`. The HIP row remains
/// unconfirmed on hardware; these targets have no portable composition proof.
/// WGSL combines the erfc and Gaussian terms before halving, so the normal
/// result at `-13` does not depend on a subnormal half-erfc intermediate.
/// Tail selection follows final reference normality, including the last
/// normal-result input. Below `-13`, a scaled Gaussian composer avoids
/// subnormal erfc/Gaussian intermediates. Source/reference checks do not
/// establish device or portable accuracy. CUDA and HIP retain
/// their IEEE tail sweep.
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
            wgsl_inputs(&main),
            wgsl_inputs(&root),
            wgsl_inputs(&mid),
            wgsl_inputs(&far),
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

#[cfg(test)]
mod tests {
    use super::*;
    use hephaestus_core::UnaryValue;

    #[test]
    fn wgsl_sweep_keeps_directed_normal_gelu_gradient_tails() {
        let inputs = wgsl_inputs(&far_inputs());
        for x in [
            -12.6,
            -13.0,
            -13.1,
            -13.3,
            f32::from_bits(0xc155_7858),
            f32::from_bits(0xc155_7859),
        ] {
            assert!(GeluGradOp::apply(f64::from(x)).abs() >= f64::from(f32::MIN_POSITIVE));
            assert!(
                inputs.contains(&x),
                "normal GELU-gradient tail removed at {x}"
            );
        }
        let next = f32::from_bits(0xc155_785a);
        assert!(GeluGradOp::apply(f64::from(next)).abs() < f64::from(f32::MIN_POSITIVE));
        assert!(!inputs.contains(&next));
    }

    #[test]
    fn wgsl_selection_preserves_existing_gelu_gradient_coverage_and_specials() {
        for inputs in [main_inputs(), root_inputs(), mid_inputs(), far_inputs()] {
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
}
