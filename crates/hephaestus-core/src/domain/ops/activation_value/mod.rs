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
//! the reference's own ULP. Each candidate threshold (a 0.001 grid near the
//! crossover) is scored in `f32`, every input evaluated, over the fixed
//! windows quoted on each constant and over sliding windows of width 0.1
//! (step 0.005) and 0.04 (step 0.0025) whose centres span ±1 around the
//! input at the switch; a window fails when the selection's maximum exceeds
//! the smaller of the always-direct and always-independent maxima over that
//! window. The fixed windows are also sampled in `f64` with 2,000,000
//! uniform inputs for each of five seeds. For `SiluGrad`, `GeluTanhGrad`
//! and `MishGrad` the per-input profiles of the two forms alternate over a
//! band near the crossover, so a narrow `f64` window can favour either form
//! depending on the seed; an `f64` excess smaller than the seed-to-seed
//! spread of the per-seed maxima in that window is sampling spread, not a
//! regression.
//!
//! Every transcendental call is eunomia's own `FloatElement`/`RealField`
//! method (ADR 0061 Decision 8); no formula below hand-rolls a function
//! eunomia already provides.

mod gated;
mod piecewise;
mod rectifier;
mod sigmoidal;
mod special;
mod stable;

#[cfg(test)]
mod tests;
