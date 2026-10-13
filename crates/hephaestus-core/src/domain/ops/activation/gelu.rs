//! Exact-GELU and GELU-gradient expressions.

use super::super::UnaryExpr;
use super::super::{GeluGradOp, GeluOp};
use crate::domain::dialect::{CudaC, Wgsl};

macro_rules! wgsl_gelu_erf_expr {
    () => {
        "(sign((x * 0.7071067811865476)) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) - 1.453152027) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) - 0.284496736) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476))))) * exp(-(x * 0.7071067811865476) * (x * 0.7071067811865476))))"
    };
}

// Single-precision `erfc` rationals for the exact GELU's negative arm
// (KERNEL-TAIL): the Sun/FreBSD `erff` `RA`/`SA` (`1.25 <= w < 1/0.35`)
// and `RB`/`SB` (`w >= 1/0.35`) tables, transcribed verbatim — the
// decimals round-trip to Sun's f32 bits; only the exponent leading
// zeros are stripped (`e-03` to `e-3`) and `e+0` dropped, both
// WGSL-canonical spellings of identical values. `s = 1/w^2` is formed
// as `2/x^2` directly: routing through `w = -x/√2` first would amplify
// its rounding by erfc's condition number `2w` (`2.2e-5` at `x = -12`,
// over budget alone), while the `x`-route keeps every sensitivity
// `O(1)` and the only large-magnitude rounding the `exp` argument's
// own f32 quantum. Likewise `/w` is spelled `-√2/x`.
macro_rules! wgsl_gelu_s {
    () => {
        "(2.0 / (x * x))"
    };
}
macro_rules! wgsl_gelu_ra_r {
    () => {
        concat!(
            "(-9.8649440333e-3) + ",
            wgsl_gelu_s!(),
            " * ((-6.9385856390e-1) + ",
            wgsl_gelu_s!(),
            " * ((-1.0558626175e+1) + ",
            wgsl_gelu_s!(),
            " * ((-6.2375331879e+1) + ",
            wgsl_gelu_s!(),
            " * ((-1.6239666748e+2) + ",
            wgsl_gelu_s!(),
            " * ((-1.8460508728e+2) + ",
            wgsl_gelu_s!(),
            " * ((-8.1287437439e+1) + ",
            wgsl_gelu_s!(),
            " * (-9.8143291473)",
            "))))))",
        )
    };
}
macro_rules! wgsl_gelu_ra_s {
    () => {
        concat!(
            "(1.0) + ",
            wgsl_gelu_s!(),
            " * ((1.9651271820e+1) + ",
            wgsl_gelu_s!(),
            " * ((1.3765776062e+2) + ",
            wgsl_gelu_s!(),
            " * ((4.3456588745e+2) + ",
            wgsl_gelu_s!(),
            " * ((6.4538726807e+2) + ",
            wgsl_gelu_s!(),
            " * ((4.2900814819e+2) + ",
            wgsl_gelu_s!(),
            " * ((1.0863500214e+2) + ",
            wgsl_gelu_s!(),
            " * ((6.5702495575) + ",
            wgsl_gelu_s!(),
            " * (-6.0424413532e-2)",
            ")))))))",
        )
    };
}
macro_rules! wgsl_gelu_rb_r {
    () => {
        concat!(
            "(-9.8649431020e-3) + ",
            wgsl_gelu_s!(),
            " * ((-7.9928326607e-1) + ",
            wgsl_gelu_s!(),
            " * ((-1.7757955551e+1) + ",
            wgsl_gelu_s!(),
            " * ((-1.6063638306e+2) + ",
            wgsl_gelu_s!(),
            " * ((-6.3756646729e+2) + ",
            wgsl_gelu_s!(),
            " * ((-1.0250950928e+3) + ",
            wgsl_gelu_s!(),
            " * (-4.8351919556e+2)",
            ")))))",
        )
    };
}
macro_rules! wgsl_gelu_rb_s {
    () => {
        concat!(
            "(1.0) + ",
            wgsl_gelu_s!(),
            " * ((3.0338060379e+1) + ",
            wgsl_gelu_s!(),
            " * ((3.2579251099e+2) + ",
            wgsl_gelu_s!(),
            " * ((1.5367296143e+3) + ",
            wgsl_gelu_s!(),
            " * ((3.1998581543e+3) + ",
            wgsl_gelu_s!(),
            " * ((2.5530502930e+3) + ",
            wgsl_gelu_s!(),
            " * ((4.7452853394e+2) + ",
            wgsl_gelu_s!(),
            " * (-2.2440952301e+1)",
            "))))))",
        )
    };
}
macro_rules! wgsl_gelu_erfc {
    (ra) => {
        concat!(
            "exp(((",
            wgsl_gelu_ra_r!(),
            ") / (",
            wgsl_gelu_ra_s!(),
            ") - 0.5625) - (x * x) * 0.5) * (-1.4142135623730951 / x)",
        )
    };
    (rb) => {
        concat!(
            "exp(((",
            wgsl_gelu_rb_r!(),
            ") / (",
            wgsl_gelu_rb_s!(),
            ") - 0.5625) - (x * x) * 0.5) * (-1.4142135623730951 / x)",
        )
    };
}

// Exact GELU (KERNEL-TAIL): the naive `1 + erf(x/√2)` transcription
// collapses to `0` below `x = -5` (`erf` rounds to `-1`, losing the
// `erfc` value, a normal float down to `x = -12`). The C spelling calls
// the native `erfc` builtin (unsuffixed, so lane promotion keeps f64
// served). The WGSL spelling keeps the Abramowitz-Stegun `erf` above
// `x = -2` (accurate to `3.3e-6` there) and takes the `erfc` rationals
// below, with the `RA`/`RB` crossover at `x = -4.0406` (`-√2/0.35`):
// `0.5·x·exp(R/S - 0.5625 - x²/2)·(-√2/x)`. The nested selects and
// f32-transcribed coefficients report `SUPPORTS_F64 = false` per the
// `Expm1Op` precedent. The deepest probe rides the `exp` argument's
// own f32 quantum (deterministic `≈5.4e-6` at `x = -12`) plus the
// driver's `exp` ULP — validated on-device, inside the clause's `1e-5`.
impl UnaryExpr<Wgsl> for GeluOp {
    const EXPR: &'static str = concat!(
        "select(select(0.5 * x * ",
        wgsl_gelu_erfc!(rb),
        ", 0.5 * x * ",
        wgsl_gelu_erfc!(ra),
        ", x > -4.0406), (0.5 * x * (1.0 + ",
        wgsl_gelu_erf_expr!(),
        ")), x > -2.0)"
    );
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for GeluOp {
    const EXPR: &'static str = "0.5f * x * erfc(-x * 0.7071067811865476f)";
}
// Exact GELU gradient (KERNEL-TAIL): the naive `0.5·(1 + erf)` term
// collapses to `0` on the negative tail, missing the `Φ` term (`~1%`
// of the total on the probes). The C spelling calls the native `erfc`
// builtin (unsuffixed builtins, so lane promotion keeps f64 served).
// The WGSL spelling reuses `GeluOp`'s `erfc`/`erf` arm structure
// verbatim for the `Φ` term — zero new coefficients — and adds the
// Gaussian term once outside the select (it needs no correction: its
// decay is the true asymptotic). Nested selects and f32-transcribed
// coefficients report `SUPPORTS_F64 = false` per the `GeluOp`
// precedent.
impl UnaryExpr<Wgsl> for GeluGradOp {
    const EXPR: &'static str = concat!(
        "select(select(0.5 * ",
        wgsl_gelu_erfc!(rb),
        ", 0.5 * ",
        wgsl_gelu_erfc!(ra),
        ", x > -4.0406), 0.5 * (1.0 + ",
        wgsl_gelu_erf_expr!(),
        "), x > -2.0) + x * exp(-0.5 * x * x) * 0.3989422804014327"
    );
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for GeluGradOp {
    const EXPR: &'static str =
        "0.5f * erfc(-x * 0.7071067811865476f) + x * exp(-0.5f * x * x) * 0.3989422804014327f";
}
