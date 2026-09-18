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
//! - At `x = ±∞` several of the formulations above form `0 · ∞`: `sigmoid`
//!   and `tanh(softplus(·))` saturate to exactly `0`/`1` at the infinite
//!   argument, so the surrounding `x · (…)` or `x / (1 + |x|)` multiplies or
//!   divides an infinity by that saturated `0`, producing `NaN` where the
//!   analytic limit is finite (`−0`, `1`, or `0`). [`SiluOp`], [`MishOp`],
//!   [`GeluOp`] and [`GeluTanhOp`] special-case `±∞` to return `x`'s sign as
//!   `±0`; their gradients and [`SoftsignOp`] special-case it to the
//!   analytic limit directly (`1`/`0`, or `copysign(1, x)`), via the local
//!   `is_infinite` helper (ADR 0061 Decision 6).
//! - [`GeluTanhGradOp`] additionally guards the finite range where `x²`
//!   itself overflows (`|x| ≳ 5e19` in `f32`): once the inner sigmoid has
//!   saturated (`s · (1 − s) == 0`), the vanished second term is never
//!   formed, so it cannot multiply that `0` by an overflowing `x²`.
//! - [`SiluGradOp`], [`GeluTanhGradOp`] and [`MishGradOp`] each compute a
//!   `1 − sigmoid(w)`-shaped companion (`1 − tanh(sp)²` for `MishGradOp`),
//!   and [`TanhGradOp`] computes `1 − y²`, choosing between two forms by
//!   region. The direct form subtracts the already-rounded value from `1`
//!   and is the more accurate one while that value is not close to `1`;
//!   the independent form (`sigmoid(−w)`, the hyperbolic
//!   `sech²(sp) = 4u/(1+u)²` with `u = exp(−2·sp)`, or the factored
//!   `(1 − y)(1 + y)`) never subtracts from `1` but pays for its own
//!   additional roundings. Each operator switches at its own measured
//!   crossover — `SILU_GRAD_CROSSOVER`, `GELU_TANH_GRAD_CROSSOVER`,
//!   `MISH_GRAD_CROSSOVER` and `TANH_GRAD_CROSSOVER` — not at a
//!   closed-form cancellation bound: `1 − sqrt(EPSILON)` ("half the
//!   significand lost") ignores the growing coefficient (`x` for `SiluGrad`,
//!   `x²` for `GeluTanhGrad`) that multiplies the companion and amplifies
//!   its rounding long before `sigmoid(w)` itself nears `1`.
//!
//! The crossover measurements share one method. The reference for an `f64`
//! input is a double-double (about 106-bit) evaluation of the operator's
//! defining formula; for an `f32` input it is that formula evaluated in
//! `f64`, whose own rounding is about `2^-29` of an `f32` ULP per operation.
//! The error of each form is its distance from that reference in units of
//! the reference's own ULP. Every `f32` input of a window is evaluated
//! (exhaustive), and each `f64` window is sampled with 2,000,000
//! uniform inputs for each of five seeds. For every candidate threshold on a
//! 0.01 grid (0.0025 for `TanhGrad`), the maximum error of the resulting
//! selection is computed per window, and the threshold is placed where the
//! windows around the switch attain their minimum while no window exceeds
//! the smaller of the two single-form maxima. For `SiluGrad`, `GeluTanhGrad`
//! and `MishGrad` the per-input profiles of the two forms interleave over a
//! band near the crossover, so the maxima quoted on those constants are
//! ties within that band rather than a sharp optimum; a difference of a few
//! thousandths of an ULP between seeds is sampling spread.
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

/// Whether `x` is `+∞` or `-∞`.
///
/// `NumericElement` exposes `is_finite`/`is_nan` but no direct
/// `is_infinite` (checked against the locked eunomia `673b8e4`); a value
/// that is neither finite nor `NaN` is exactly an infinity.
#[must_use]
fn is_infinite<T: RealField>(x: T) -> bool {
    !x.is_finite() && !x.is_nan()
}

/// Largest `x` for which [`SiluGradOp`] forms `1 − sigmoid(x)` by direct
/// subtraction; above it, `stable_sigmoid(−x)`.
///
/// Measured by the method in the module documentation. Per input, direct
/// is the more accurate form for every `x` up to about `2.6` and the
/// independent form for every `x` above about `2.8`; between the two the
/// forms alternate. Thresholds from `2.50` to `2.61` keep every window at or
/// below the smaller single-form maximum in all six sample sets (`f32` and
/// five `f64` seeds); `2.6` lies inside that range. Window maxima, `f32` /
/// largest `f64` seed, at the previous threshold `2` → at `2.6`: `[1.5, 2.5]`
/// 1.885 / 1.902 → 1.678 / 1.721 ULP (always-direct: 1.678 / 1.721),
/// `[1.9, 2.1]` 1.885 / 1.894 → 1.366 / 1.381, `[2, 8]` 1.885 / 1.894 →
/// 1.871 / 1.894 (always-independent: 1.885 / 1.894); `[0, 2]` stays
/// 2.085 / 2.051 and `[8, 30]` 1.710 / 1.717.
const SILU_GRAD_CROSSOVER: f64 = 2.6;

/// Largest `w = 2z` for which [`GeluTanhGradOp`] forms `1 − sigmoid(w)` by
/// direct subtraction; above it, `stable_sigmoid(−w)`.
///
/// Measured by the method in the module documentation. Per input, direct
/// is the more accurate form for every `w` up to about `2.35` and the
/// independent form for every `w` above about `2.5`; between the two the
/// forms alternate. `2.43` is the only grid value within 0.01 ULP of the
/// smaller single-form maximum in every window and sample set; its largest
/// excess, 0.004 ULP in one `f64` seed of `[1, 1.4]`, is below that window's
/// seed-to-seed spread (1.61 to 1.78 ULP). Window maxima, `f32` / largest
/// `f64` seed, at the previous threshold `2` → at `2.43`: `[1, 1.4]` 1.852
/// / 1.780 → 1.685 / 1.779 ULP, `[1.15, 1.25]` 1.742 / 1.771 → 1.338 /
/// 1.460; `[1.2, 1.6]` stays 1.852 / 1.841, `[1.4, 2]` 1.795 / 1.831,
/// `[0, 2]` 1.973 / 1.829 and `[2, 8]` 1.600 / 1.595.
const GELU_TANH_GRAD_CROSSOVER: f64 = 2.43;

/// Largest `sp = softplus(x)` for which [`MishGradOp`] forms `1 − tanh(sp)²`
/// by direct subtraction; above it, `sech²(sp)`.
///
/// The switch is on `sp`, not on `t = tanh(sp)`: `softplus(0) = ln 2`, so
/// `t` already exceeds `½` at `x = 0`, and every threshold on `sp` below
/// about `0.71` routes the neighbourhood of `x = 0`, where direct is the
/// more accurate form, to `sech²`.
///
/// Measured by the method in the module documentation. Per input, direct
/// is the more accurate form for every `sp` up to about `1.51` and `sech²`
/// for every `sp` above about `1.61`; between the two the forms alternate.
/// Thresholds from `1.17` to `1.61` keep every window at or below the
/// smaller single-form maximum in all six sample sets, and `1.49` to `1.60`
/// additionally minimise the `[1, 1.6]` window; `1.55` lies at the centre.
/// Window maxima, `f32` / largest `f64` seed, at the previous threshold `2`
/// → at `1.55`: `[1.6, 2.1]` 1.553 / 1.559 → 1.190 / 1.194 ULP (always-sech²:
/// 1.190 / 1.194), `[1.8, 1.9]` 1.553 / 1.594 → 1.120 / 1.152, `[1, 1.6]`
/// 1.396 / 1.466 → 1.246 / 1.243; `[0.5, 1]` stays 1.528 / 1.480, `[0, 2]`
/// 2.355 / 2.321 and `[2, 8]` 1.049 / 1.024.
const MISH_GRAD_CROSSOVER: f64 = 1.55;

/// Largest `|y|` for which [`TanhGradOp`] forms `1 − y²` directly; above it,
/// `(1 − y)(1 + y)`.
///
/// Measured by the method in the module documentation. The two forms do
/// not interleave: per input, direct has the smaller maximum error for
/// every `|y|` up to `0.75` (at most 0.750 ULP below `√½` and 1.000 ULP
/// from `√½` to `0.75`, where the factored form reaches 1.085), and the
/// factored form for every `|y|` above `0.7525` (below 1.000 ULP up to
/// `√¾`, where direct rises to 2.000). Thresholds from `0.75` to `0.865` keep
/// every window at or below the smaller single-form maximum; `0.75` is the
/// crossover itself. Window maxima, `f32` / largest `f64` seed, at the
/// previous threshold `0.5` → at `0.75`: `[0.45, 0.55]` 1.000 / 1.000 →
/// 0.750 / 0.750 ULP, `[0.6, 0.8]` 1.085 / 1.085 → 1.000 / 1.000,
/// `[0.5, 1]` 1.085 / 1.085 → 1.035 / 1.033; `[0, 0.5]` stays 0.625 / 0.625
/// and `[0.9, 1]` 1.015 / 1.016.
const TANH_GRAD_CROSSOVER: f64 = 0.75;

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

/// `max(x, 0)` via an explicit `is_nan` check rather than `T::max` alone:
/// eunomia's `max_scalar` contract has a single `NaN` operand *ignored*, so
/// `NaN.max(0) == 0` — silently discarding the `NaN` instead of propagating
/// it, the opposite of IEEE 754's own `maxNum`/comparison-predicate
/// semantics for a unary activation.
impl UnaryValue for ReluOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            x
        } else {
            x.max(<T as NumericElement>::ZERO)
        }
    }
}

/// Takes the input (ADR 0061 Decision 7), matching the WGSL/CUDA rendering.
/// `x > 0` is `false` for `NaN` (IEEE 754 unordered comparisons), so the
/// naive two-arm form silently returns `0` for a `NaN` input instead of
/// propagating it; the explicit `is_nan` check comes first.
impl UnaryValue for ReluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            x
        } else if x > <T as NumericElement>::ZERO {
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
/// The two branches keep different multiplication orders on purpose. The
/// direct branch multiplies `s` and `1 - s` into the product one factor at a
/// time (`two * x * s * (1 - s) * c0 * (…)`); near the derivative's root
/// (`x ≈ -0.7525`), where the sum cancels, pre-grouping `s * (1 - s)` there
/// changes individual results by hundreds of ULP of the result (at
/// `x = -0.75311023_f32`: 0.016 ULP in this order, 512 ULP pre-grouped,
/// against the double-double reference). The independent branch
/// pre-groups `saturation = s * sigmoid(-w)`, the order its measured window
/// maxima on `GELU_TANH_GRAD_CROSSOVER` were taken with.
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

/// `clamp` (`min_scalar`/`max_scalar`) has the same "a single `NaN` operand
/// is ignored" contract `max_scalar` does (see [`ReluOp`]), so
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Derivation: `f64`'s `exp`/`ln_1p` are accurate to a few ULP, so a
    /// `1e-6` *relative* tolerance bounds every assertion below far above
    /// that rounding noise while catching a cancellation collapse to `0`
    /// outright (a collapse is 100% relative error). An *absolute* `1e-6`
    /// tolerance is unsound here: `softplus(-20) ≈ 2.06e-9` and
    /// `mish(-20) ≈ -4.12e-8` are themselves smaller than `1e-6`, so an
    /// absolute bound passes exactly the collapse-to-zero defect these two
    /// tests exist to catch (the test integrity gap ADR 0061 Decision 6
    /// exists to close).
    const REL_TOL: f64 = 1e-6;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= REL_TOL * b.abs()
    }

    #[test]
    fn softplus_matches_the_cancellation_and_overflow_cases() {
        // softplus(-20) = ln(1 + exp(-20)); exp(-20) = 2.061e-9 is far below
        // f64 ULP(1), so ln(1+e) ~ e to full relative precision, giving the
        // reference 2.061153622e-9 (ADR 0061 Decision 6). A collapse to
        // exactly 0 (the defect this test guards) is 100% relative error,
        // caught by the relative bound above regardless of the reference's
        // own tiny magnitude.
        assert!(close(softplus_value(-20.0f64), 2.061_153_622e-9,));
        // softplus(100) = 100 + ln_1p(exp(-100)); exp(-100) underflows f64's
        // ability to move 100.0 at all (ULP(100) ~ 1.4e-14 >> exp(-100)).
        assert_eq!(softplus_value(100.0f64), 100.0);
    }

    /// Mutation-killing regression: a mutant substituting the naive
    /// `ln(1 + exp(x))` for the `ln_1p`-based form collapses to exactly `0`
    /// here. `exp(-100) ≈ 3.72e-44` is far below `f64`'s `ulp(1) ≈
    /// 1.11e-16`, so `1.0 + exp(-100)` rounds to exactly `1.0` and
    /// `ln(1.0) == 0.0` — 100% relative error, which the module's own
    /// `REL_TOL = 1e-6` test above (at `x = -20`, where `exp(-20) ≈
    /// 2.06e-9` is still far above `f64`'s ULP) would already catch, but
    /// this test exercises the collapse at the much more extreme magnitude
    /// the mutation search actually found. `softplus(x) = x + ln(1+e^x)`
    /// for the negated branch gives `softplus(-100) = ln(1+e^-100) =
    /// e^-100 - e^-200/2 + O(e^-300)`; the dropped `O(e^-200)` term is
    /// about 40 orders of magnitude below `1e-12` relative to the leading
    /// `e^-100` term, so `softplus(-100)` must match `exp(-100)` to far
    /// tighter than `1e-12` relative — a bound the naive mutant's exact-`0`
    /// output fails outright.
    #[test]
    fn softplus_negative_100_matches_exp_within_1e_minus_12_relative() {
        let got = softplus_value(-100.0f64);
        let reference = (-100.0f64).exp();
        let relative_error = (got - reference).abs() / reference.abs();
        assert!(
            relative_error < 1e-12,
            "softplus(-100) = {got:e}, exp(-100) = {reference:e}, relative error {relative_error:e}"
        );
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
