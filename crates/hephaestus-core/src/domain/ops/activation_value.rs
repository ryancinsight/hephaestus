//! Value functions for the activation-family unary markers (ADR 0061).
//!
//! These formulations are chosen to stay a normal, nonzero float wherever the
//! true mathematical value is one, even in tails where a direct transcription
//! of the dialect expression cancels to zero (ADR 0061 Decision 6):
//!
//! - [`SoftplusOp`] computes `max(x, 0) + ln_1p(exp(-|x|))` rather than
//!   `ln(1 + exp(x))`, so the negative tail (`softplus(-20) ≈ 2.06e-9`) never
//!   collapses through `ln(1 + 0)` and the positive tail
//!   (`softplus(100) == 100`) never overflows through `exp(100)`.
//! - [`SiluOp`], [`GeluTanhOp`], and the sigmoid-shaped gradients route
//!   through `stable_sigmoid`, the standard branch-selected sigmoid that
//!   evaluates `exp` on a value that cannot overflow on either branch, so
//!   `silu(-89)` stays the normal float it mathematically is rather than
//!   `x / (1 + exp(89))` overflowing `exp` to infinity and the quotient to
//!   zero.
//! - [`GeluOp`] and [`GeluGradOp`] compute `erfc(-x / √2)` — the identity
//!   `1 + erf(z) = erfc(-z)` — using eunomia's native `erfc`, which is
//!   accurate in exactly the tail where `1 + erf(z)` cancels for very
//!   negative `z`.
//! - [`MishOp`] and [`MishGradOp`] compose the same accurate
//!   `softplus_value` and `stable_sigmoid` rather than
//!   `tanh(ln(1 + exp(x)))`, so `mish(-20)` reproduces the reference
//!   `-4.12e-8` instead of cancelling to zero.
//!
//! Every transcendental call is eunomia's own `FloatElement`/`RealField`
//! method (ADR 0061 Decision 8); no formula below hand-rolls a function
//! eunomia already provides.

use super::{
    EluGradOp, EluOp, ErfOp, ErfcOp, GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp,
    HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, LgammaOp, MishGradOp, MishOp,
    ReluGradOp, ReluOp, SigmoidGradOp, SigmoidOp, SiluGradOp, SiluOp, SoftplusGradOp, SoftplusOp,
    SoftsignGradOp, SoftsignOp, TanhGradOp, TanhOp, UnaryValue,
};
use eunomia::{NumericElement, RealField};

/// Numerically stable logistic sigmoid `1 / (1 + exp(-x))`.
///
/// Evaluating `exp(-x)` directly overflows for very negative `x` (`exp(89)`
/// already exceeds `f32::MAX`), collapsing the quotient to `0` where the true
/// value is a tiny but normal float. Branching on the sign evaluates `exp` at
/// an argument that can never overflow: `exp(-x)` for `x ≥ 0` and `exp(x)` for
/// `x < 0`, the textbook stable form.
#[must_use]
pub(crate) fn stable_sigmoid<T: RealField>(x: T) -> T {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    if x >= zero {
        one / (one + (-x).exp())
    } else {
        let e = x.exp();
        e / (one + e)
    }
}

/// Numerically stable softplus `ln(1 + exp(x))`, computed as
/// `max(x, 0) + ln_1p(exp(-|x|))` (ADR 0061 Decision 6): the positive branch
/// never overflows `exp`, and `ln_1p` keeps the negative tail's cancellation
/// away from `ln`.
#[must_use]
pub(crate) fn softplus_value<T: RealField>(x: T) -> T {
    x.max(<T as NumericElement>::ZERO) + (-<T as NumericElement>::abs(x)).exp().ln_1p()
}

impl UnaryValue for ErfOp {
    fn apply<T: RealField>(x: T) -> T {
        x.erf()
    }
}

impl UnaryValue for ErfcOp {
    fn apply<T: RealField>(x: T) -> T {
        x.erfc()
    }
}

impl UnaryValue for LgammaOp {
    fn apply<T: RealField>(x: T) -> T {
        x.lgamma()
    }
}

impl UnaryValue for ReluOp {
    fn apply<T: RealField>(x: T) -> T {
        x.max(<T as NumericElement>::ZERO)
    }
}

/// Takes the input (ADR 0061 Decision 7), matching the WGSL/CUDA rendering.
impl UnaryValue for ReluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

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

/// Takes the forward output `y = tanh(x)` (ADR 0061 Decision 7).
impl UnaryValue for TanhGradOp {
    fn apply<T: RealField>(y: T) -> T {
        <T as NumericElement>::ONE - y * y
    }
}

/// `elu(x) = x` for `x ≥ 0`, `expm1(x)` for `x < 0` — `expm1` rather than
/// `exp(x) - 1` keeps the negative branch accurate near zero.
impl UnaryValue for EluOp {
    fn apply<T: RealField>(x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            x.exp_m1()
        }
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7).
impl UnaryValue for EluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            x.exp()
        }
    }
}

/// `0.5 · x · erfc(-x / √2)`, the identity `1 + erf(z) = erfc(-z)` applied to
/// the textbook `0.5 · x · (1 + erf(x / √2))` (ADR 0061 Decision 6): `erfc` is
/// accurate exactly where `1 + erf` would cancel to zero for very negative
/// `x`.
impl UnaryValue for GeluOp {
    fn apply<T: RealField>(x: T) -> T {
        let inv_sqrt_2 = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let half = T::from_f64(0.5);
        half * x * (-(x * inv_sqrt_2)).erfc()
    }
}

/// Takes the input (ADR 0061 Decision 7). Same `erfc` identity as
/// [`GeluOp`] for the error-function term; the Gaussian term
/// `x · exp(-x²/2)` needs no correction — its decay to zero for very negative
/// `x` is the true asymptotic, not a cancellation artifact.
impl UnaryValue for GeluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let inv_sqrt_2 = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let inv_sqrt_2pi = T::from_f64(0.398_942_280_401_432_7);
        let half = T::from_f64(0.5);
        let neg_half = T::from_f64(-0.5);
        half * (-(x * inv_sqrt_2)).erfc() + x * (neg_half * x * x).exp() * inv_sqrt_2pi
    }
}

/// `x · sigmoid(2z)`, `z = √(2/π)·(x + 0.044715·x³)` — the identity
/// `0.5·(1 + tanh(z)) = sigmoid(2z)` applied to the textbook
/// `0.5·x·(1 + tanh(z))` (ADR 0061 Decision 6): `stable_sigmoid` never
/// forms `1 + tanh(z)` directly, which loses precision once `tanh(z)` rounds
/// to `±1`.
impl UnaryValue for GeluTanhOp {
    fn apply<T: RealField>(x: T) -> T {
        let z = gelu_tanh_arg(x);
        x * stable_sigmoid(z + z)
    }
}

/// Takes the input (ADR 0061 Decision 7); not in the forward-output list, so
/// it recomputes from `x` like the WGSL/CUDA rendering. Rewritten in terms of
/// `s = sigmoid(2z)` using `1 + tanh(z) = 2s` and
/// `1 - tanh(z)² = 4s(1 - s)`, so the derivative
/// `0.5·(1+tanh(z)) + 0.5·x·(1-tanh(z)²)·z'(x)` never forms `1 + tanh(z)`
/// directly.
impl UnaryValue for GeluTanhGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let c0 = T::from_f64(0.797_884_560_802_865_4);
        let c1 = T::from_f64(0.044715);
        let z = gelu_tanh_arg(x);
        let s = stable_sigmoid(z + z);
        let one = <T as NumericElement>::ONE;
        let two = one + one;
        let three = two + one;
        s + two * x * s * (one - s) * c0 * (one + three * c1 * x * x)
    }
}

/// `√(2/π)·(x + 0.044715·x³)`, the argument of the tanh-approximated GELU's
/// inner `tanh`, shared by [`GeluTanhOp`] and [`GeluTanhGradOp`].
fn gelu_tanh_arg<T: RealField>(x: T) -> T {
    let c0 = T::from_f64(0.797_884_560_802_865_4);
    let c1 = T::from_f64(0.044715);
    c0 * (x + c1 * x * x * x)
}

/// `x · sigmoid(x)` via `stable_sigmoid` (ADR 0061 Decision 6): the naive
/// `x / (1 + exp(-x))` overflows `exp(-x)` to infinity for `x` around `-89`,
/// where the true value (`≈ -1.75e-37`) is still a normal float.
impl UnaryValue for SiluOp {
    fn apply<T: RealField>(x: T) -> T {
        x * stable_sigmoid(x)
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7).
impl UnaryValue for SiluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let sig = stable_sigmoid(x);
        sig * (<T as NumericElement>::ONE + x * (<T as NumericElement>::ONE - sig))
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

/// `x · tanh(softplus(x))`, composing `softplus_value` (ADR 0061
/// Decision 6): `softplus(x)` never underflows to exactly `0` in the negative
/// tail, and `tanh` is well-conditioned near its own zero, so the composition
/// reproduces `mish(-20) ≈ -4.12e-8` rather than cancelling to zero.
impl UnaryValue for MishOp {
    fn apply<T: RealField>(x: T) -> T {
        x * softplus_value(x).tanh()
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7). Reuses the same accurate
/// `softplus_value` and `stable_sigmoid` building blocks as [`MishOp`].
impl UnaryValue for MishGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let t = softplus_value(x).tanh();
        t + x * (<T as NumericElement>::ONE - t * t) * stable_sigmoid(x)
    }
}

impl UnaryValue for HardsigmoidOp {
    fn apply<T: RealField>(x: T) -> T {
        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;
        let six = T::from_f64(6.0);
        let half = T::from_f64(0.5);
        (x / six + half).clamp(zero, one)
    }
}

impl UnaryValue for HardsigmoidGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let three = T::from_f64(3.0);
        if x > -three && x < three {
            <T as NumericElement>::ONE / T::from_f64(6.0)
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

impl UnaryValue for HardswishOp {
    fn apply<T: RealField>(x: T) -> T {
        let zero = <T as NumericElement>::ZERO;
        let three = T::from_f64(3.0);
        let six = T::from_f64(6.0);
        x * (x + three).clamp(zero, six) / six
    }
}

impl UnaryValue for HardswishGradOp {
    fn apply<T: RealField>(x: T) -> T {
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

impl UnaryValue for SoftsignOp {
    fn apply<T: RealField>(x: T) -> T {
        x / (<T as NumericElement>::ONE + <T as NumericElement>::abs(x))
    }
}

impl UnaryValue for SoftsignGradOp {
    fn apply<T: RealField>(x: T) -> T {
        let denom = <T as NumericElement>::ONE + <T as NumericElement>::abs(x);
        <T as NumericElement>::ONE / (denom * denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Derivation: `f64`'s `exp`/`ln_1p` are accurate to a few ULP, so an
    /// `f64` reference compared at `1e-6` absolute tolerance bounds every
    /// assertion below far above rounding noise while staying scale-free for
    /// these O(1)-to-O(100) magnitudes.
    const TOL: f64 = 1e-6;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL
    }

    #[test]
    fn softplus_matches_the_cancellation_and_overflow_cases() {
        // softplus(-20) = ln(1 + exp(-20)); exp(-20) = 2.061e-9 is far below
        // f64 ULP(1), so ln(1+e) ~ e to full relative precision, giving the
        // reference 2.061153622e-9 (ADR 0061 Decision 6).
        assert!(close(softplus_value(-20.0f64), 2.061_153_622e-9,));
        // softplus(100) = 100 + ln_1p(exp(-100)); exp(-100) underflows f64's
        // ability to move 100.0 at all (ULP(100) ~ 1.4e-14 >> exp(-100)).
        assert_eq!(softplus_value(100.0f64), 100.0);
    }

    #[test]
    fn stable_sigmoid_avoids_the_overflow_collapse() {
        // sigmoid(-89) = 1/(1+exp(89)); exp(89) overflows f32 (> 3.4e38), so
        // the naive quotient rounds to 0 exactly. The true value, exp(-89) to
        // leading order, is a normal f32 (~1.97e-39 is representable; the
        // silu product below lands in the exponent range that matters).
        let s = stable_sigmoid(-89.0f32);
        assert!(s > 0.0, "sigmoid(-89) must be a nonzero normal float");
        let silu = -89.0f32 * s;
        assert!(silu != 0.0, "silu(-89) must not underflow to zero");
    }

    #[test]
    fn mish_reproduces_the_negative_tail_reference() {
        // mish(-20) = -20 * tanh(softplus(-20)); tanh is linear near 0, so
        // tanh(2.061e-9) ~ 2.061e-9, giving -4.122e-8 (ADR 0061 Decision 6:
        // "Mish(-20) 0 against -4.12e-8" names this as the rendering defect
        // the value function must not repeat).
        let m = MishOp::apply(-20.0f64);
        assert!(close(m, -4.122_307_244e-8));
    }

    #[test]
    fn gelu_matches_the_erfc_identity_in_the_negative_tail() {
        // gelu(-10) = 0.5 * -10 * erfc(10/sqrt(2)); erfc(7.07) is a tiny but
        // normal double (~7.6e-13), so gelu(-10) must be a nonzero normal
        // float, never the zero a naive 1+erf(-z) cancellation would give.
        let g = GeluOp::apply(-10.0f64);
        assert!(g != 0.0, "gelu(-10) must be a nonzero normal float");
        assert!(g < 0.0, "gelu(-10) must keep the sign of a negative input");
    }

    #[test]
    fn round_trips_the_gelu_reference_at_zero_and_large_positive() {
        assert_eq!(GeluOp::apply(0.0f64), 0.0);
        // gelu(x) -> x as x -> +inf: erfc(-large) -> 2, so 0.5*x*2 = x.
        assert!(close(GeluOp::apply(10.0f64), 10.0));
    }
}
