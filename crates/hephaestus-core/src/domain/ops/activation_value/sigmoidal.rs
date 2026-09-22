//! Value functions for the saturating sigmoidal family: logistic sigmoid,
//! `tanh`, softplus and softsign, forward and gradient.

use super::super::{
    SigmoidGradOp, SigmoidOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, TanhGradOp,
    TanhOp, UnaryValue,
};
use super::stable::{TANH_GRAD_CROSSOVER, is_infinite, softplus_value, stable_sigmoid};
use eunomia::{NumericElement, RealField};

impl UnaryValue for SigmoidOp {
    fn apply<T: RealField>(x: T) -> T {
        stable_sigmoid(x)
    }
}

/// Takes the forward output `y = sigmoid(x)` (ADR 0061 Decision 7): the
/// argument passed to [`Self::apply`] is already `sigmoid`'s result.
impl UnaryValue for SigmoidGradOp {
    fn apply<T: RealField>(y: T) -> T {
        y * (<T as NumericElement>::ONE - y)
    }
}

impl UnaryValue for TanhOp {
    fn apply<T: RealField>(x: T) -> T {
        x.tanh()
    }
}

/// Takes the forward output `y = tanh(x)` (ADR 0061 Decision 7). For small
/// `|y|`, `1 - y*y` rounds twice (the square, then the difference) while
/// `(1 - y) * (1 + y)` rounds three times, and the direct form is the more
/// accurate one (at most 0.625 ULP against the factored form's 1.5 for
/// `|y| ≤ ½`). As `|y| → 1` the direct form subtracts a rounded square from
/// a nearly equal `1` and loses the significand (0.99988 in `f32`: 989 ULP;
/// `1 - 6.7e-9` in `f64`: about 2.7e7 ULP), while the factored form stays
/// within about 1 ULP. The switch sits at the measured crossover
/// `TANH_GRAD_CROSSOVER`.
impl UnaryValue for TanhGradOp {
    fn apply<T: RealField>(y: T) -> T {
        let one = <T as NumericElement>::ONE;
        if <T as NumericElement>::abs(y) <= T::from_f64(TANH_GRAD_CROSSOVER) {
            one - y * y
        } else {
            (one - y) * (one + y)
        }
    }
}

impl UnaryValue for SoftplusOp {
    fn apply<T: RealField>(x: T) -> T {
        softplus_value(x)
    }
}

/// The softplus derivative is the logistic sigmoid.
impl UnaryValue for SoftplusGradOp {
    fn apply<T: RealField>(x: T) -> T {
        stable_sigmoid(x)
    }
}

impl UnaryValue for SoftsignOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `1 + |x|` is also infinite at `x = ±∞`, so the quotient forms
            // `∞ / ∞`; the analytic limit is `±1`.
            return <T as NumericElement>::ONE.copysign(x);
        }
        x / (<T as NumericElement>::ONE + <T as NumericElement>::abs(x))
    }
}

impl UnaryValue for SoftsignGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let denom = <T as NumericElement>::ONE + <T as NumericElement>::abs(x);
        <T as NumericElement>::ONE / (denom * denom)
    }
}
