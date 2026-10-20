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

use super::{SweepDialect, assert_unary_sweep_within, below_cap, sweep_inputs, without_subnormals};

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
            8.9, // WGSL cap: results go subnormal past here
            9.0,
            9.19,
            9.194_549_f32.next_down(),
            10.0, // deep tail interior (CUDA/HIP only)
        ],
    )
}

fn wgsl_far_inputs(inputs: &[f32]) -> Vec<f32> {
    below_cap(&without_subnormals(inputs), 8.9)
}

/// Sweep [`ErfcOp`](hephaestus_core::ErfcOp) on one backend.
///
/// Bounds: CUDA `5` ULP (`erfcf` at `4` per the current Programming Guide
/// table, but an exhaustive third-party scan measures 4.49 ULP on CUDA 11
/// and the vendor table is generated from non-exhaustive tests, so the
/// bound carries margin past both); HIP `4` ULP (the CUDA row carried
/// unconfirmed — no HIP-published `erfcf` row was found; any excess on
/// hardware CI is a finding); WGSL `32` ULP (the Chebyshev form at its
/// published `1.2e-7` fractional bound — about `2` ULP — plus Horner
/// evaluation rounding, the `t` division at `2.5`, and the driver's `exp`
/// over arguments reaching `-10` at the `z = 3` handoff. The `exp` term
/// dominates and varies by driver: lavapipe measures `18.0` ULP worst at
/// `x = 2.19`, Metal `17.1` at `x = 2.84`, local Vulkan at most `16` —
/// all inside the WGSL `(3 + 2|x|)` `exp` envelope (`14` and `21` at the
/// two failing arguments), so the drivers conform and the old `16` bound
/// was under-margined. `32` carries the band's margin convention over the
/// cross-driver measured worst, as the far band does). The positive tail
/// sweeps in two
/// graded bands: mid `128` ULP and far `256` ULP on WGSL (worst-case
/// envelope `z²`-rounding plus driver-`exp` plus evaluation: the analytic
/// composition ceiling sits just under `128` at `z = 6` (`36` from `z²`
/// rounding, `9` Horner, `2` form, `77` spec-`exp`, `1` division, `0.5`
/// product — every term at max at once, which never co-occurs), and
/// lavapipe measures `89.2` worst at `x = 5.81`, inside it; about `131`
/// at the `8.9` cap, where the driver measures 126 ULP worst at
/// `x = 8.81`). `128` covers the measured worst with margin and the
/// ceiling barely: a driver nearer the spec `exp` envelope would force
/// `256`, so any excess is a finding. CUDA `5` and HIP `4` ULP on both (the native
/// builtin holds its row across the bands — it forms no `z²`). WGSL
/// additionally caps the far band at `x = 8.9`, where the true result goes
/// subnormal and flushing is permitted; CUDA and HIP sweep the tail as
/// IEEE. The C dialects already render native `erfc`, so their clauses pin
/// rather than fix.
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
        for x in [9.0, 9.19] {
            assert!(ErfcOp::apply(f64::from(x)) >= f64::from(f32::MIN_POSITIVE));
            assert!(inputs.contains(&x), "normal erfc tail removed at {x}");
        }
        assert!(inputs.iter().any(|x| x.is_nan()));
        assert!(inputs.contains(&f32::INFINITY));
        assert!(inputs.contains(&f32::NEG_INFINITY));
    }
}
