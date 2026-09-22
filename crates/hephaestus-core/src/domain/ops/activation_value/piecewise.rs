//! Value functions for the piecewise-linear approximations: hard sigmoid and
//! hard swish, forward and gradient.

use super::super::{HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, UnaryValue};
use eunomia::{NumericElement, RealField};

/// `clamp` (`min_scalar`/`max_scalar`) has the same "a single `NaN` operand
/// is ignored" contract `max_scalar` does (see [`ReluOp`](crate::ReluOp)), so
/// `NaN.clamp(0, 1)` silently returns a bound instead of `NaN` — the
/// explicit `is_nan` check comes first.
impl UnaryValue for HardsigmoidOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;
        let six = T::from_f64(6.0);
        let half = T::from_f64(0.5);
        (x / six + half).clamp(zero, one)
    }
}

/// `x > -3 && x < 3` is `false` for `NaN`, so the naive two-arm form
/// silently returns `0` for a `NaN` input instead of propagating it; the
/// explicit `is_nan` check comes first.
impl UnaryValue for HardsigmoidGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let three = T::from_f64(3.0);
        if x > -three && x < three {
            <T as NumericElement>::ONE / T::from_f64(6.0)
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Divides the clamp by `6` before multiplying by `x` (`x * (clamp / 6)`
/// rather than `x * clamp / 6`): for `x ≥ 3` the clamp saturates to `6` and
/// the true value is `x` exactly, but `x * 6` overflows to `inf` for `x`
/// past `f32::MAX / 6 ≈ 5.67e37` (`f64::MAX / 6 ≈ 3.0e307`) before the `/ 6`
/// would bring it back down; dividing first keeps every intermediate value
/// at most `1`. For `x ≤ -3` the clamp instead saturates to `0`, and the
/// value is `0` exactly, but at `x = -∞` the literal `x * (0 / 6)` forms
/// `-∞ · 0 = NaN`; short-circuiting to `-0` avoids ever forming that
/// product, and matches the rendering's `-0` sign convention for negative
/// `x` in the finite case too.
impl UnaryValue for HardswishOp {
    fn apply<T: RealField>(x: T) -> T {
        let zero = <T as NumericElement>::ZERO;
        let three = T::from_f64(3.0);
        if x <= -three {
            return -zero;
        }
        let six = T::from_f64(6.0);
        x * ((x + three).clamp(zero, six) / six)
    }
}

/// The three-arm `x >= 3` / `x > -3` / else form is `false`/`false` for
/// `NaN`, falling through to the final `else` and silently returning `0`
/// instead of propagating `NaN`; the explicit `is_nan` check comes first.
impl UnaryValue for HardswishGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let one = <T as NumericElement>::ONE;
        let two = one + one;
        let three = two + one;
        let six = T::from_f64(6.0);
        if x >= three {
            one
        } else if x > -three {
            (two * x + three) / six
        } else {
            <T as NumericElement>::ZERO
        }
    }
}
