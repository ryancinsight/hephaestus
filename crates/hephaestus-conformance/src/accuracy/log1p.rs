//! Accuracy sweep for [`Log1pOp`](hephaestus_core::Log1pOp).
//!
//! The failing range is the cancellation zone around zero, where `log(1 + x)`
//! loses every significant bit; the sweep covers it densely, the `x = -1`
//! pole and its floating-point neighbourhood, huge arguments (where the old
//! regrouped form overflowed), and the structural specials.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, Log1pOp, UnaryExpr};

use super::{SweepDialect, assert_unary_sweep_within, sweep_inputs, without_subnormals};

/// 4096 points over `[-0.5, 0.5]` plus 1024 over `[-1, 10]`, the pole
/// neighbourhood, huge arguments, and form-switch edges.
fn inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate set is
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -0.5,
        0.5,
        4096,
        &[
            1.0e-8,
            -1.0e-8, // total cancellation under the old form
            1.0e-7,
            -1.0e-7, //
            2.0f32.powi(-24),
            -2.0f32.powi(-24), // 1+x rounds to 1 below here
            1.0e-40,
            -1.0e-40, // subnormal inputs
            0.0,      // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(
        -1.0,
        10.0,
        1024,
        &[
            -1.0,                    // the pole: -inf
            -1.0 + 2.0f32.powi(-24), // first float above the pole
            -0.999_999_9,            // near-pole interior
            -2.0,                    // NaN domain
            -1.0e30,                 // NaN domain interior
            0.5,
            1.0,
            2.0,                        // log-table knees through u = 1 + x
            0.414_213_56,               // sqrt(2) - 1: shift-up/core handoff
            -0.292_893_22,              // sqrt(2)/2 - 1: core/shift-down handoff
            -0.5,                       // shift-down/direct handoff
            core::f32::consts::E - 1.0, // u = e
            10.0,
            100.0,  // moderate tail
            1.0e30, // huge but finite u
            1.0e37, // the ADR's overflow case for unguarded forms
        ],
    ));
    inputs
}

/// Sweep [`Log1pOp`](hephaestus_core::Log1pOp) on one backend.
///
/// Bounds: CUDA `1` ULP (Programming Guide Table 17: `log1pf` 1 ULP full
/// range); HIP `1` ULP (HIP math API: `log1pf` 1 ULP on `[-0.9, 10]`, carried
/// outside that tested range by the shared ocml implementation — any excess
/// there is a finding); WGSL `6` ULP (musl `log1pf` k = 0 core with the
/// published `2^-34.24` approximation bound over the middle, exact shifts
/// into it from the abutting bands, direct `log(1 + x)` only where the
/// argument sits under the relative part of the spec table; the measured
/// Vulkan maximum sits well inside with margin for cross-driver variance,
/// and DX12/Metal/HIP execute the same sweep in hardware CI).
pub fn assert_log1p_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Log1pOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let max_ulp = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => 1.0,
        SweepDialect::Wgsl => 6.0,
    };
    let inputs = inputs();
    // WGSL permits flushing subnormals; the other dialects sweep them.
    let inputs = if dialect == SweepDialect::Wgsl {
        without_subnormals(&inputs)
    } else {
        inputs
    };
    assert_unary_sweep_within::<D, E, Log1pOp>(device, ops, "log1p", &inputs, max_ulp);
}
