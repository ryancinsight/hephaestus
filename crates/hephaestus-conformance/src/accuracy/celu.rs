//! Accuracy sweep for [`CeluOp`](hephaestus_core::CeluOp).
//!
//! `celu(x) = x` for `x ≥ 0` and `α · expm1(x / α)` below, so the failing
//! range scales with `α`: the sweep covers `[-4α, 4α]` densely per `α` over
//! five `α` values spanning subnormal-adjacent to wide, plus `α`-scaled
//! tail edges. Gradient operations are out of scope: `CeluGradOp` renders
//! `exp(x / α)` with no cancellation class.

use hephaestus_core::{
    CeluOp, ComputeDevice, ParameterizedUnaryExpr, ParameterizedUnaryOps, ParameterizedUnaryValue,
};

use super::{SweepDialect, assert_parameterized_sweep_within, sweep_inputs, without_subnormals};

/// The `α` values swept: degenerate zero, negative, subnormal-adjacent,
/// fractional, unit, and wide. Zero and negative `α` pin the seam's
/// arithmetic at the degenerate inputs both backends must agree on.
const ALPHAS: [f32; 7] = [0.0, -1.0, 0.125, 0.5, 1.0, 2.0, 8.0];

/// 1024 points over `[-4α, 4α]` plus `α`-scaled tail edges. At `α = 0` the
/// dense range collapses to a point and the edges carry the sweep.
fn inputs(alpha: f32) -> Vec<f32> {
    sweep_inputs(
        -4.0 * alpha,
        4.0 * alpha,
        1024,
        &[
            0.0, // +0 kink (the builder adds -0)
            4.0 * alpha,
            8.0 * alpha, // positive identity interior
            -8.0 * alpha,
            -88.0 * alpha,  // negative tail
            -104.0 * alpha, // exp vanishes: the branch is -α
            -0.3 * alpha,
            0.3 * alpha, // Taylor/direct handoff, α-scaled
            -1.0e30,     // deep tail
            1.0e-8,
            -1.0e-8, // cancellation at the kink
            1.0e-40,
            -1.0e-40, // subnormal inputs
        ],
    )
}

/// Sweep [`CeluOp`](hephaestus_core::CeluOp) on one backend.
///
/// Bounds derive from the `α · expm1(x / α)` composition. CUDA `5` ULP:
/// `/` at `2` (Table 17 without `-prec-div`), `expm1f` at `1` with unit
/// input sensitivity near zero rising modestly along the tail, `*` at `0.5`,
/// plus margin. HIP `6` ULP: `expm1f` at `1`, `/` carried at `≤ 2`
/// (unconfirmed against AMD's divider; any excess is a finding), `*` at
/// `0.5`, plus margin. WGSL `10` ULP: `/` at `2.5` (accuracy table) feeding
/// the Taylor/`exp` core with unit input sensitivity on the negative branch
/// (the `z · E′(z) / E(z)` factor peaks at 1 near the kink and vanishes
/// down the tail), `*` correctly rounded, plus margin for cross-driver
/// variance.
pub fn assert_celu_sweep<D, P>(device: &D, ops: &P, dialect: SweepDialect)
where
    D: ComputeDevice,
    P: ParameterizedUnaryOps<D>,
    CeluOp: ParameterizedUnaryExpr<P::Dialect> + ParameterizedUnaryValue,
{
    dialect.check::<P::Dialect>();
    let max_ulp = match dialect {
        SweepDialect::CudaC => 5.0,
        SweepDialect::HipC => 6.0,
        SweepDialect::Wgsl => 10.0,
    };
    for alpha in ALPHAS {
        let case = inputs(alpha);
        // WGSL permits flushing subnormals; the other dialects sweep them.
        let case = if dialect == SweepDialect::Wgsl {
            without_subnormals(&case)
        } else {
            case
        };
        assert_parameterized_sweep_within::<D, P, CeluOp>(
            device,
            ops,
            "celu",
            [alpha, 0.0],
            &case,
            max_ulp,
        );
    }
}
