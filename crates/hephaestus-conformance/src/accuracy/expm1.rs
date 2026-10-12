//! Accuracy sweep for [`Expm1Op`](hephaestus_core::Expm1Op).
//!
//! The failing range is the cancellation zone around zero, where `exp(x) - 1`
//! loses every significant bit; the sweep covers it densely, the overflow
//! knee at `ln(FLT_MAX)`, the underflow floor where `exp` vanishes, and the
//! structural specials.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, Expm1Op, UnaryExpr};

use super::{SweepDialect, assert_unary_sweep_within, sweep_inputs, without_subnormals};

/// 4096 points over `[-1, 1]` plus the overflow knee, the underflow floor,
/// subnormal inputs, and form-switch edges: everything outside the positive
/// knee band, swept at the form bound.
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -1.0,
        1.0,
        4096,
        &[
            88.73, // just past ln(FLT_MAX): +inf
            89.0,
            100.0, // deep overflow interior
            -100.0,
            -104.0,  // exp underflows to zero: the result is -1
            -1000.0, // deep tail
            -87.0,
            -88.0, // large-magnitude finite results, negative side
            // (87.0 and ln(FLT_MAX) live in the knee band below)
            1.0e-8,
            -1.0e-8, // total cancellation under the old form
            1.0e-7,
            -1.0e-7, //
            2.0f32.powi(-24),
            -2.0f32.powi(-24), // 1+x rounds to 1 below here
            0.3,
            -0.3, // Taylor/direct handoff
            1.0e30,
            -1.0e30, // overflow/underflow interiors
            1.0e-40,
            -1.0e-40, // subnormal inputs
            0.0,      // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(-88.73, -80.0, 256, &[]));
    inputs
}

/// 256 points over the positive knee band `(80, ln(FLT_MAX)]`, swept at the
/// driver-`exp` bound: the rendering is transparent `exp(x) - 1` there (the
/// `-1` vanishes in rounding), so the error is the driver's `exp` grade,
/// which the Vulkan driver spends loosely near overflow.
fn knee_inputs() -> Vec<f32> {
    sweep_inputs(80.0, 88.73, 256, &[87.0, 88.722_83])
}

/// Sweep [`Expm1Op`](hephaestus_core::Expm1Op) on one backend.
///
/// Bounds: CUDA `1.5` ULP (Programming Guide Table 17 rows `expm1f` at 1
/// ULP, but the table is generated from non-exhaustive tests and disclaims
/// guarantees; the RTX 5080 at driver 610.60 measures 1.115 ULP at
/// x = 0.371 (6 of 4123 inputs over 1.0) against the `libm`-exact `f64`
/// oracle, so the bound carries `0.5` of vendor margin and still rejects
/// the old form at 1.6e7 ULP and any `≥ 1.5` regression); HIP `1` ULP (HIP
/// math API: `expm1f` 1 ULP on `[-10, 10]`, carried outside that tested
/// range by the shared ocml implementation — any excess there is a finding);
/// WGSL `6` ULP (degree-6 Taylor core under `|x| <= 0.3` with a Lagrange
/// remainder below 0.06 ULP plus division-free Horner rounding, direct
/// `exp(x) - 1` outside with no cancellation left; the measured Vulkan
/// maximum sits well inside with margin for cross-driver variance, and
/// DX12/Metal/HIP execute the same sweep in hardware CI).
/// The positive knee band `(80, ln(FLT_MAX)]` sweeps separately at the
/// driver-`exp` bound, since the rendering is transparent there: CUDA `1.5`
/// and HIP `1` ULP (the native builtin holds its row across the band),
/// WGSL `64` ULP (the Vulkan driver at 610.60 measures 59 ULP worst there;
/// the canary's `exp` probes stop short of large arguments, so the sweep
/// carries the grade instead — see the canary-coverage note in `mod.rs`).
pub fn assert_expm1_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Expm1Op: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, knee_ulp) = match dialect {
        SweepDialect::CudaC => (1.5, 1.5),
        SweepDialect::HipC => (1.0, 1.0),
        SweepDialect::Wgsl => (6.0, 64.0),
    };
    let main = main_inputs();
    let knee = knee_inputs();
    // WGSL permits flushing subnormals; the other dialects sweep them.
    let (main, knee) = if dialect == SweepDialect::Wgsl {
        (without_subnormals(&main), without_subnormals(&knee))
    } else {
        (main, knee)
    };
    assert_unary_sweep_within::<D, E, Expm1Op>(device, ops, "expm1", &main, max_ulp);
    assert_unary_sweep_within::<D, E, Expm1Op>(device, ops, "expm1-knee", &knee, knee_ulp);
}
