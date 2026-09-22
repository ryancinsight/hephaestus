//! Value functions for the self-gated family: GELU (both the exact `erfc`
//! form and the `tanh` approximation), SiLU and Mish, forward and gradient.

use super::super::{
    GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp, MishGradOp, MishOp, SiluGradOp, SiluOp,
    UnaryValue,
};
use super::stable::{
    GELU_TANH_GRAD_CROSSOVER, MISH_GRAD_CROSSOVER, SILU_GRAD_CROSSOVER, gelu_tanh_arg, is_infinite,
    softplus_value, stable_sigmoid,
};
use eunomia::{NumericElement, RealField};

/// `0.5 · x · erfc(-x / √2)`, the identity `1 + erf(z) = erfc(-z)` applied to
/// the textbook `0.5 · x · (1 + erf(x / √2))` (ADR 0061 Decision 6): `erfc` is
/// accurate exactly where `1 + erf` would cancel to zero for very negative
/// `x`.
impl UnaryValue for GeluOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `erfc(∓∞) = 0`/`2` exactly, so `half * x * erfc(…)` forms
            // `∓∞ · 0` at `x = -∞` (the `+∞` branch multiplies by `2`, never
            // cancelling); the analytic limit is `x` itself.
            return if x.is_sign_positive() {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
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
        if is_infinite(x) {
            // The `erfc` term saturates to `1`/`0` at `x = ±∞`, but the
            // Gaussian term `x · exp(-x²/2) · c` forms `±∞ · 0`; the
            // analytic limit of the whole derivative is `1` at `+∞`, `0` at
            // `-∞`.
            return if x.is_sign_positive() {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
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
        if is_infinite(x) {
            // `sigmoid(2z)` saturates to `1`/`0` at `x = ±∞` (`z → ±∞`), so
            // `x * sigmoid(2z)` forms `-∞ · 0` at `-∞`; the analytic limit
            // is `x` itself.
            return if x.is_sign_positive() {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        let z = gelu_tanh_arg(x);
        x * stable_sigmoid(z + z)
    }
}

/// Takes the input (ADR 0061 Decision 7); not in the forward-output list, so
/// it recomputes from `x` like the WGSL/CUDA rendering. Rewritten in terms of
/// `s = sigmoid(2z)` using `1 + tanh(z) = 2s` and
/// `1 - tanh(z)² = 4s(1 - s)`, so the derivative
/// `0.5·(1+tanh(z)) + 0.5·x·(1-tanh(z)²)·z'(x)` never forms `1 + tanh(z)`
/// directly. `1 - s` is formed by direct subtraction for
/// `w = 2z ≤` `GELU_TANH_GRAD_CROSSOVER` and as `sigmoid(-w)` above it.
///
/// Each branch keeps the multiplication order of the single-form evaluation
/// it is measured against: the direct branch multiplies `s` and `1 - s` into
/// the product one factor at a time (`two * x * s * (1 - s) * c0 * (…)`),
/// the independent branch pre-groups `saturation = s * sigmoid(-w)`. The
/// two orders are equally accurate on average but round differently at
/// individual inputs, and the other order per branch raises the `f32` window
/// maxima measured by the method in the module documentation: pre-grouping
/// the direct branch leaves 249 of 1,211 sliding windows above the smaller
/// single-form maximum (largest excess 0.152 ULP, 1.936 against 1.784 over
/// `[0.515, 0.555]`), and ungrouping the independent branch leaves 210 of
/// 1,571 (0.093 ULP, 1.774 against 1.681 over `[1.625, 1.725]`). Within
/// `±0.05` of the derivative's root near `x = -0.7525` the pre-grouped order
/// would have the smaller maximum (4.473 against 4.681 ULP of the dominant
/// term `s` in `f32`), but that gain does not carry over the rest of the
/// direct region.
impl UnaryValue for GeluTanhGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `s` saturates to `1`/`0` at `x = ±∞`, so the vanished second
            // term would form `x² · 0` against an infinite `x²`; the
            // analytic limit of the derivative is `1` at `+∞`, `0` at `-∞`.
            return if x.is_sign_positive() {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let c0 = T::from_f64(0.797_884_560_802_865_4);
        let c1 = T::from_f64(0.044715);
        let z = gelu_tanh_arg(x);
        let w = z + z;
        let s = stable_sigmoid(w);
        let one = <T as NumericElement>::ONE;
        let two = one + one;
        let three = two + one;
        if w <= T::from_f64(GELU_TANH_GRAD_CROSSOVER) {
            let one_minus_s = one - s;
            if s * one_minus_s == <T as NumericElement>::ZERO {
                // `s` has saturated to exactly `0` or `1` — this includes
                // the finite range where `x * x` below overflows
                // (`|x| ≳ 5e19` in `f32`): the second term's true limit is
                // `0`, but forming it would multiply that vanished
                // `s(1-s)` by an overflowing `x²` (`0 · ∞ = NaN`). Return
                // the saturated `s` directly instead.
                return s;
            }
            s + two * x * s * one_minus_s * c0 * (one + three * c1 * x * x)
        } else {
            let one_minus_s = stable_sigmoid(-w);
            let saturation = s * one_minus_s;
            if saturation == <T as NumericElement>::ZERO {
                return s;
            }
            s + two * x * saturation * c0 * (one + three * c1 * x * x)
        }
    }
}

/// `x · sigmoid(x)` via `stable_sigmoid` (ADR 0061 Decision 6): the naive
/// `x / (1 + exp(-x))` overflows `exp(-x)` to infinity for `x` around `-89`,
/// where the true value (`≈ -1.75e-37`) is still a normal float.
impl UnaryValue for SiluOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `stable_sigmoid(-∞) == 0` exactly, so `x * sigmoid(x)` forms
            // `-∞ · 0`; the analytic limit is `x` itself.
            return if x.is_sign_positive() {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        x * stable_sigmoid(x)
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7). `1 - sigmoid(x)` is
/// formed by direct subtraction for `x ≤` `SILU_GRAD_CROSSOVER` and as
/// `stable_sigmoid(-x)` above it.
impl UnaryValue for SiluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `sig` saturates to `1`/`0` at `x = ±∞`, so `x * (1 - sig)`
            // forms `±∞ · 0`; the analytic limit of the derivative is `1`
            // at `+∞`, `0` at `-∞`.
            return if x.is_sign_positive() {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let sig = stable_sigmoid(x);
        let one_minus_sig = if x <= T::from_f64(SILU_GRAD_CROSSOVER) {
            <T as NumericElement>::ONE - sig
        } else {
            stable_sigmoid(-x)
        };
        sig * (<T as NumericElement>::ONE + x * one_minus_sig)
    }
}

/// `x · tanh(softplus(x))`, composing `softplus_value` (ADR 0061
/// Decision 6): `softplus(x)` never underflows to exactly `0` in the negative
/// tail, and `tanh` is well-conditioned near its own zero, so the composition
/// reproduces `mish(-20) ≈ -4.12e-8` rather than cancelling to zero.
impl UnaryValue for MishOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `softplus(-∞) == 0` so `tanh(softplus(-∞)) == 0`, and
            // `x * tanh(softplus(x))` forms `-∞ · 0`; the analytic limit is
            // `x` itself.
            return if x.is_sign_positive() {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        x * softplus_value(x).tanh()
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7). Reuses the same accurate
/// `softplus_value` and `stable_sigmoid` building blocks as [`MishOp`].
/// `1 - t²`, `t = tanh(sp)`, is formed by direct subtraction for
/// `sp ≤` `MISH_GRAD_CROSSOVER` and above it as the hyperbolic identity
/// `sech²(sp) = 4u / (1 + u)²`, `u = exp(-2·sp)`, which never subtracts from
/// `1` but rounds more often than the direct form.
impl UnaryValue for MishGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if is_infinite(x) {
            // `t` saturates to `1`/`0` at `x = ±∞`, so `x * (1 - t²) *
            // sigmoid(x)` forms `±∞ · 0`; the analytic limit of the
            // derivative is `1` at `+∞`, `0` at `-∞`.
            return if x.is_sign_positive() {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let sp = softplus_value(x);
        let t = sp.tanh();
        let one = <T as NumericElement>::ONE;
        let one_minus_t_sq = if sp <= T::from_f64(MISH_GRAD_CROSSOVER) {
            one - t * t
        } else {
            let two = one + one;
            let four = two + two;
            let u = (-two * sp).exp();
            four * u / ((one + u) * (one + u))
        };
        t + x * one_minus_t_sq * stable_sigmoid(x)
    }
}
