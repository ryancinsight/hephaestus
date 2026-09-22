//! Shared numerically stable kernels and the measured crossover thresholds.
//!
//! The two sigmoid-shaped kernels every activation below composes, the
//! infinity predicate their `0 · ∞` special cases test, and the four
//! gradient crossovers whose values the module documentation derives.

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
pub(super) fn is_infinite<T: RealField>(x: T) -> bool {
    !x.is_finite() && !x.is_nan()
}

/// Largest `x` for which [`SiluGradOp`] forms `1 − sigmoid(x)` by direct
/// subtraction; above it, `stable_sigmoid(−x)`.
///
/// Measured by the method in the module documentation. Per input, direct
/// is the more accurate form up to `x` of about `2.6` and the independent
/// form above about `2.8`; between the two the forms alternate. In `f32`,
/// thresholds from `2.626` to `2.635` leave no fixed or sliding window
/// (1,212 windows, centres `1.6` to `3.6`) above the smaller single-form
/// maximum; `2.625` leaves 37 (largest excess 0.080 ULP) and `2.636` four
/// (0.024 ULP). Window maxima, `f32` / largest of five `f64` seeds, at the
/// previous threshold `2` → at `2.63`: `[1.5, 2.5]` 1.885 / 1.902 → 1.678 /
/// 1.721 ULP, `[1.9, 2.1]` 1.885 / 1.894 → 1.366 / 1.381, `[2.55, 2.65]`
/// 1.871 / 1.904 → 1.791 / 1.901, `[2, 8]` 1.885 / 1.894 → 1.867 / 1.894;
/// `[0, 2]` stays 2.085 / 2.051 and `[8, 30]` 1.710 / 1.717. The largest
/// `f64` excess at `2.63`, 0.062 ULP in one seed of `[2.5, 2.7]`, is below
/// that window's seed-to-seed spread of the always-direct maximum (1.854 to
/// 1.948).
pub(super) const SILU_GRAD_CROSSOVER: f64 = 2.63;

/// Largest `w = 2z` for which [`GeluTanhGradOp`] forms `1 − sigmoid(w)` by
/// direct subtraction; above it, `stable_sigmoid(−w)`.
///
/// Measured by the method in the module documentation. Per input, direct
/// is the more accurate form up to `w` of about `2.35` and the independent
/// form above about `2.5`; between the two the forms alternate. In `f32`,
/// thresholds from `2.459` to `2.476` leave no fixed or sliding window
/// (1,211 windows, input centres `0.4` to `2.4`) above the smaller
/// single-form maximum; `2.458` leaves seven (largest excess 0.077 ULP)
/// and `2.477` 34 (0.111 ULP). Window maxima, `f32` / largest of five `f64`
/// seeds, at the previous threshold `2` → at `2.47`: `[1, 1.4]` 1.852 /
/// 1.780 → 1.685 / 1.779 ULP, `[1.15, 1.25]` 1.742 / 1.771 → 1.338 /
/// 1.460, `[1.38, 1.42]` 1.791 / 1.835 → 1.714 / 1.826, `[1.2, 1.6]` 1.852 /
/// 1.841 → 1.795 / 1.841; `[1.4, 2]` stays 1.795 / 1.831, `[0, 2]` 1.973 /
/// 1.829 and `[2, 8]` 1.600 / 1.595. The largest `f64` excess at `2.47`,
/// 0.063 ULP in one seed of `[1.2, 1.6]`, is below that window's
/// seed-to-seed spread of the always-independent maximum (1.752 to 1.841).
pub(super) const GELU_TANH_GRAD_CROSSOVER: f64 = 2.47;

/// Largest `sp = softplus(x)` for which [`MishGradOp`] forms `1 − tanh(sp)²`
/// by direct subtraction; above it, `sech²(sp)`.
///
/// The switch is on `sp`, not on `t = tanh(sp)`: `softplus(0) = ln 2`, so
/// `t` already exceeds `½` at `x = 0`, and every threshold on `sp` below
/// about `0.71` routes the neighbourhood of `x = 0`, where direct is the
/// more accurate form, to `sech²`.
///
/// Measured by the method in the module documentation. Per input, direct
/// is the more accurate form up to `sp` of about `1.51` and `sech²` above
/// about `1.61`; between the two the forms alternate. No threshold clears
/// every window: in `f32`, thresholds from `1.541` to `1.557` leave the
/// fewest, 4 of 1,212 windows (input centres `0.3` to `2.3`), each exceeding
/// the smaller single-form maximum by at most 0.0144 ULP (1.2208 against
/// 1.2063 over `[1.3575, 1.3975]`, where direct is better throughout but
/// covering it would move the switch past `1.62`); `1.540` leaves seven and
/// `1.558` fifteen. Window maxima, `f32` / largest of five `f64` seeds, at the
/// previous threshold `2` → at `1.55`: `[1.6, 2.1]` 1.553 / 1.559 → 1.190 /
/// 1.194 ULP, `[1.8, 1.9]` 1.553 / 1.594 → 1.120 / 1.152, `[1, 1.6]` 1.396 /
/// 1.466 → 1.246 / 1.243, `[1.26, 1.36]` 1.226 / 1.274 → 1.191 / 1.299;
/// `[0.5, 1]` stays 1.528 / 1.480, `[0, 2]` 2.355 / 2.321 and `[2, 8]`
/// 1.049 / 1.024. The largest `f64` excess at `1.55`, 0.073 ULP in one seed
/// of `[1.26, 1.36]`, is below that window's seed-to-seed spread of the
/// always-`sech²` maximum (1.238 to 1.347).
pub(super) const MISH_GRAD_CROSSOVER: f64 = 1.55;

/// Largest `|y|` for which [`TanhGradOp`] forms `1 − y²` directly; above it,
/// `(1 − y)(1 + y)`.
///
/// Measured by the method in the module documentation. Direct errs by at
/// most 0.750 ULP below `|y| = √½` and 1.000 ULP from `√½` up to `√¾`, where
/// it rises to 2.000; the factored form reaches 1.085 just above `√½` and
/// falls below 1.000 ULP near `0.75`. In `f32`, thresholds from `0.7500` to
/// `0.7525` leave no fixed or sliding window (613 windows, centres `0` to
/// `1`) above the smaller single-form maximum; `0.7495` leaves 41 (largest
/// excess 0.0005 ULP) and `0.753` one (0.005 ULP). Window maxima, `f32` /
/// largest of five `f64` seeds, at the previous threshold `0.5` → at `0.75`:
/// `[0.45, 0.55]` 1.000 / 1.000 → 0.750 / 0.750 ULP, `[0.6, 0.8]` 1.085 /
/// 1.085 → 1.000 / 1.000, `[0.5, 1]` 1.085 / 1.085 → 1.035 / 1.033;
/// `[0, 0.5]` stays 0.625 / 0.625 and `[0.9, 1]` 1.015 / 1.016. No `f64`
/// window exceeds the smaller single-form maximum at `0.75`.
pub(super) const TANH_GRAD_CROSSOVER: f64 = 0.75;

/// `√(2/π)·(x + 0.044715·x³)`, the argument of the tanh-approximated GELU's
/// inner `tanh`, shared by [`GeluTanhOp`] and [`GeluTanhGradOp`].
pub(super) fn gelu_tanh_arg<T: RealField>(x: T) -> T {
    let c0 = T::from_f64(0.797_884_560_802_865_4);
    let c1 = T::from_f64(0.044715);
    c0 * (x + c1 * x * x * x)
}
