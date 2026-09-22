//! Layout-validation and value-function regressions for this module.

use super::*;
use crate::domain::dialect::{CudaC, HipC, Wgsl};
use crate::domain::error::HephaestusError;
use eunomia::{NumericElement, RealField};
use leto::Layout;

#[test]
fn hardtanh_keeps_the_open_lower_and_upper_boundary_at_x_itself() {
    // Boundary convention (ADR 0061): `select(select(x, second, x >
    // second), first, x < first)` uses strict `<`/`>`, so `x == first`
    // or `x == second` falls through both selects to `x` itself —
    // numerically identical to the clamped bound, but the code path
    // taken differs, which the reversed-bounds case below distinguishes.
    assert_eq!(HardtanhOp::apply(-0.75_f32, -0.75, 1.25), -0.75);
    assert_eq!(HardtanhOp::apply(1.25_f32, -0.75, 1.25), 1.25);
    assert_eq!(HardtanhOp::apply(-2.0_f32, -0.75, 1.25), -0.75);
    assert_eq!(HardtanhOp::apply(2.0_f32, -0.75, 1.25), 1.25);
}

#[test]
fn hardtanh_matches_the_nested_select_order_under_reversed_bounds() {
    // A plain `x.clamp(first, second)` requires `first <= second` and
    // diverges from the nested-select rendering once `first > second`
    // (`clamp` panics in debug; a naive min/max reordering would give
    // the wrong branch here). `EXPR`'s outer select conditions on `x <
    // first` using the *original* `x`, not the inner select's result, so
    // working through `select(select(x, second, x > second), first, x <
    // first)` by hand for `first = 0.5 > second = -0.25`:
    // - x = 0.5: inner picks `second` (0.5 > -0.25); outer condition
    //   `x < first` is `0.5 < 0.5` = false, so the result is the inner
    //   value `second = -0.25`, not `first`.
    assert_eq!(HardtanhOp::apply(0.5_f32, 0.5, -0.25), -0.25);
    assert_eq!(HardtanhOp::apply(-2.0_f32, 0.5, -0.25), 0.5);
    assert_eq!(HardtanhOp::apply(2.0_f32, 0.5, -0.25), -0.25);
}

#[test]
fn hardtanh_grad_is_the_open_interval_excluding_both_endpoints() {
    assert_eq!(HardtanhGradOp::apply(-0.75_f32, -0.75, 1.25), 0.0);
    assert_eq!(HardtanhGradOp::apply(1.25_f32, -0.75, 1.25), 0.0);
    assert_eq!(HardtanhGradOp::apply(0.0_f32, -0.75, 1.25), 1.0);
}

#[test]
fn threshold_selects_the_replacement_at_and_below_the_boundary() {
    // `x > first ? x : second` (strict): `x == first` takes `second`.
    assert_eq!(ThresholdOp::apply(0.5_f32, 0.5, -3.25), -3.25);
    assert_eq!(ThresholdOp::apply(0.500_001_f32, 0.5, -3.25), 0.500_001);
}

#[test]
fn threshold_grad_matches_the_strict_boundary() {
    assert_eq!(ThresholdGradOp::apply(0.5_f32, 0.5, 0.0), 0.0);
    assert_eq!(ThresholdGradOp::apply(0.500_001_f32, 0.5, 0.0), 1.0);
}

#[test]
fn leaky_relu_takes_the_identity_branch_at_both_signed_zeros() {
    // `x >= 0.0` (non-strict): `-0.0 >= 0.0` is `true` in IEEE-754, so
    // both signed zeros take the identity branch, unlike the gradient.
    assert_eq!(LeakyReluOp::apply(0.0_f32, 0.25, 0.0), 0.0);
    assert_eq!(LeakyReluOp::apply(-0.0_f32, 0.25, 0.0), -0.0);
    assert_eq!(LeakyReluOp::apply(-1.0_f32, 0.25, 0.0), -0.25);
}

#[test]
fn leaky_relu_grad_selects_the_negative_slope_at_both_signed_zeros() {
    // `x > 0.0` (strict): `-0.0 > 0.0` and `0.0 > 0.0` are both `false`,
    // so both signed zeros take the negative-slope branch — the exact
    // case `hephaestus-conformance`'s `assert_parameterized_unary_contract`
    // pins across every backend.
    assert_eq!(LeakyReluGradOp::apply(0.0_f32, 0.25, 0.0), 0.25);
    assert_eq!(LeakyReluGradOp::apply(-0.0_f32, 0.25, 0.0), 0.25);
    assert_eq!(LeakyReluGradOp::apply(1.0_f32, 0.25, 0.0), 1.0);
}

#[test]
fn hardshrink_and_grad_use_the_strict_magnitude_threshold() {
    assert_eq!(HardshrinkOp::apply(0.5_f32, 0.5, 0.0), 0.0);
    assert_eq!(HardshrinkOp::apply(-0.5_f32, 0.5, 0.0), 0.0);
    assert_eq!(HardshrinkOp::apply(0.500_001_f32, 0.5, 0.0), 0.500_001);
    assert_eq!(HardshrinkGradOp::apply(0.5_f32, 0.5, 0.0), 0.0);
    assert_eq!(HardshrinkGradOp::apply(0.500_001_f32, 0.5, 0.0), 1.0);
}

#[test]
fn softshrink_and_grad_use_the_strict_threshold_on_both_sides() {
    assert_eq!(SoftshrinkOp::apply(0.5_f32, 0.5, 0.0), 0.0);
    assert_eq!(SoftshrinkOp::apply(-0.5_f32, 0.5, 0.0), 0.0);
    // Compared against `x - first`/`x + first` computed the same way the
    // implementation computes it, not a separately-rounded literal: the
    // two nearby f32 values subtract exactly (Sterbenz's lemma), but an
    // independently parsed `0.000_001_f32` literal rounds to a different
    // nearest float than that exact difference.
    let just_above = 0.500_001_f32;
    assert_eq!(SoftshrinkOp::apply(just_above, 0.5, 0.0), just_above - 0.5);
    let just_below = -0.500_001_f32;
    assert_eq!(SoftshrinkOp::apply(just_below, 0.5, 0.0), just_below + 0.5);
    assert_eq!(SoftshrinkGradOp::apply(0.5_f32, 0.5, 0.0), 0.0);
    assert_eq!(SoftshrinkGradOp::apply(0.500_001_f32, 0.5, 0.0), 1.0);
    assert_eq!(SoftshrinkGradOp::apply(-0.500_001_f32, 0.5, 0.0), 1.0);
}

#[test]
fn celu_uses_exp_m1_to_avoid_negative_tail_cancellation() {
    // celu(x) = alpha * expm1(x / alpha) for x < 0. At x = -1e-8, alpha
    // = 1, exp(-1e-8) rounds to exactly 1.0 in f32 (1e-8 is below
    // f32::EPSILON ~= 1.19e-7), so a direct `exp(x) - 1` transcription
    // cancels to 0.0 exactly, losing the tail entirely (ADR 0061
    // Decision 6). `exp_m1` computes it to a few ULP of the true value,
    // which is `x` to first order (the `x^2/2` term is ~5e-17,
    // negligible at f32 precision) — bound derived as 8 ULP at this
    // magnitude for eunomia's `exp_m1` accuracy plus rounding headroom.
    let value = CeluOp::apply(-1.0e-8_f32, 1.0, 0.0);
    assert!(value != 0.0, "celu(-1e-8) must not cancel to zero");
    let tolerance = 8.0 * f32::EPSILON * 1.0e-8_f32;
    assert!(
        (value - (-1.0e-8_f32)).abs() <= tolerance,
        "celu(-1e-8) = {value}, expected ~-1e-8 within {tolerance}"
    );
}

#[test]
fn celu_matches_x_at_and_above_zero_and_the_exp_branch_below() {
    assert_eq!(CeluOp::apply(0.0_f32, 0.5, 0.0), 0.0);
    assert_eq!(CeluOp::apply(2.0_f32, 0.5, 0.0), 2.0);
    assert_eq!(CeluGradOp::apply(0.0_f32, 0.5, 0.0), 1.0);
    // Compared against `(x / first).exp()` via the same `RealField::exp`
    // the implementation calls (eunomia's `libm`-routed `exp`, not
    // necessarily bit-identical to `std::f32::exp`).
    assert_eq!(
        CeluGradOp::apply(-1.0_f32, 0.5, 0.0),
        (-1.0_f32 / 0.5).exp()
    );
}

#[test]
fn value_functions_are_generic_over_every_shipped_real_field() {
    // Generic Instantiation Coverage (standards): one check function,
    // instantiated across every `RealField` eunomia ships today
    // (f32/f64), rather than a per-type test copy.
    fn check<T: RealField>() {
        let two = T::from_f64(2.0);
        let neg_one = T::from_f64(-1.0);
        let one = T::from_f64(1.0);
        let half = T::from_f64(0.5);
        let zero = <T as NumericElement>::ZERO;

        assert_eq!(HardtanhOp::apply(two, neg_one, one), one);
        assert_eq!(HardtanhOp::apply(-two, neg_one, one), neg_one);
        assert_eq!(ThresholdOp::apply(two, one, neg_one), two);
        assert_eq!(ThresholdOp::apply(zero, one, neg_one), neg_one);
        assert_eq!(LeakyReluOp::apply(-two, half, zero), -two * half);
        assert_eq!(HardshrinkOp::apply(-two, one, zero), -two);
        assert_eq!(HardshrinkOp::apply(half, one, zero), zero);
        assert_eq!(SoftshrinkOp::apply(two, one, zero), one);
        assert_eq!(CeluOp::apply(two, one, zero), two);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn expressions_pin_parameter_and_boundary_conventions() {
    assert_eq!(
        <HardtanhOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
        "select(select(x, second, x > second), first, x < first)"
    );
    assert_eq!(
        <HardtanhGradOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
        "select(0.0, 1.0, (x > first) && (x < second))"
    );
    assert_eq!(
        <ThresholdOp as ParameterizedUnaryExpr<CudaC>>::EXPR,
        "x > first ? x : second"
    );
    assert_eq!(
        <ThresholdGradOp as ParameterizedUnaryExpr<HipC>>::EXPR,
        "x > first ? 1.0 : 0.0"
    );
    assert_eq!(
        <LeakyReluOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
        "select(first * x, x, x >= 0.0)"
    );
    assert_eq!(
        <LeakyReluGradOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
        "select(first, 1.0, x > 0.0)"
    );
    assert_eq!(
        <LeakyReluGradOp as ParameterizedUnaryExpr<CudaC>>::EXPR,
        "x > 0.0f ? 1.0f : first"
    );
    assert_eq!(
        <LeakyReluGradOp as ParameterizedUnaryExpr<HipC>>::EXPR,
        "x > 0.0f ? 1.0f : first"
    );
    assert_eq!(
        <SoftshrinkOp as ParameterizedUnaryExpr<CudaC>>::EXPR,
        "x > first ? x - first : (x < -first ? x + first : 0.0f)"
    );
    assert_eq!(
        <CeluGradOp as ParameterizedUnaryExpr<HipC>>::EXPR,
        "x >= 0.0f ? 1.0f : expf(x / first)"
    );
}

#[test]
fn writable_layout_validation_accepts_injective_interleaving() {
    let layout = Layout::try_new([2, 3], [3, 2], 0).expect("valid test layout");
    let len = validate_parameterized_output(&layout, 8).expect("injective layout");
    assert_eq!(len, 6);
}

#[test]
fn writable_layout_validation_rejects_nonzero_stride_aliasing() {
    let layout = Layout::try_new([2, 2], [1, 1], 0).expect("valid test layout");
    assert!(matches!(
        validate_parameterized_output(&layout, 3),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output layout must be non-overlapping"
    ));
}

#[test]
fn writable_layout_validation_uses_bounded_sparse_fallback() {
    let injective = Layout::try_new([2, 3], [300, 200], 0).expect("valid test layout");
    let len = validate_parameterized_output(&injective, 701).expect("injective sparse layout");
    assert_eq!(len, 6);

    let overlapping = Layout::try_new([2, 3], [300, 150], 0).expect("valid test layout");
    assert!(matches!(
        validate_parameterized_output(&overlapping, 601),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output layout must be non-overlapping"
    ));
}
