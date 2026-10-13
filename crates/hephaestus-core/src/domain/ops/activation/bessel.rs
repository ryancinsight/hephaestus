//! Bessel J0/J1/K0 expressions.

use super::super::UnaryExpr;
use super::super::{J0Op, J1Op, K0Op};
use crate::domain::dialect::{CudaC, HipC, Wgsl};

// Bessel J0/J1 share one structure across dialects: the Numerical-Recipes
// rational approximation for `|x| < 8` and Hankel's asymptotic expansion
// otherwise, with the same coefficients and nesting as leto's scalar
// `j0`/`j1`. Only the absolute-value spelling differs (`abs` vs `fabs`),
// so the branch macros take it as `$ax`; `sin`/`cos`/`sqrt` stay unsuffixed
// so C overload resolution picks the lane width, as `SinOp` does.
macro_rules! bessel_y2 {
    () => {
        "((x) * (x))"
    };
}

macro_rules! bessel_z {
    ($ax:literal) => {
        concat!("(8.0 / (", $ax, "))")
    };
}

macro_rules! bessel_yh {
    ($ax:literal) => {
        concat!("((8.0 / (", $ax, ")) * (8.0 / (", $ax, ")))")
    };
}

macro_rules! bessel_xx {
    ($ax:literal, $off:literal) => {
        concat!("((", $ax, ") - ", $off, ")")
    };
}

macro_rules! j0_rational {
    () => {
        concat!(
            "((",
            "57568490574.0 + ",
            bessel_y2!(),
            "*(-13362590354.0 + ",
            bessel_y2!(),
            "*(651619640.7 + ",
            bessel_y2!(),
            "*(-11214424.18 + ",
            bessel_y2!(),
            "*(77392.33017 + ",
            bessel_y2!(),
            "*-184.9052456))))",
            ") / (",
            "57568490411.0 + ",
            bessel_y2!(),
            "*(1029532985.0 + ",
            bessel_y2!(),
            "*(9494680.718 + ",
            bessel_y2!(),
            "*(59272.64853 + ",
            bessel_y2!(),
            "*(267.8532712 + ",
            bessel_y2!(),
            "))))",
            "))"
        )
    };
}

macro_rules! j0_hankel {
    ($ax:literal) => {
        concat!(
            "(sqrt(2.0 / (3.141592653589793 * (",
            $ax,
            "))) * ((",
            "1.0 + ",
            bessel_yh!($ax),
            "*(-0.001098628627 + ",
            bessel_yh!($ax),
            "*(0.000002734510407 + ",
            bessel_yh!($ax),
            "*(-2.073370639e-6 + ",
            bessel_yh!($ax),
            "*2.093887211e-7)))",
            ") * cos(",
            bessel_xx!($ax, "0.7853981633974483"),
            ") - (",
            bessel_z!($ax),
            ") * (",
            "-0.01562499995 + ",
            bessel_yh!($ax),
            "*(0.0001430488765 + ",
            bessel_yh!($ax),
            "*(-6.911147651e-5 + ",
            bessel_yh!($ax),
            "*(7.621095161e-5 - ",
            bessel_yh!($ax),
            "*9.34935152e-7)))",
            ") * sin(",
            bessel_xx!($ax, "0.7853981633974483"),
            ")))"
        )
    };
}

macro_rules! j1_rational {
    () => {
        concat!(
            "(((x) * (72362614232.0 + ",
            bessel_y2!(),
            "*(-7895059235.0 + ",
            bessel_y2!(),
            "*(242396853.1 + ",
            bessel_y2!(),
            "*(-2972611.439 + ",
            bessel_y2!(),
            "*(15704.48260 + ",
            bessel_y2!(),
            "*-30.16036606))))))) / (",
            "144725228442.0 + ",
            bessel_y2!(),
            "*(2300535178.0 + ",
            bessel_y2!(),
            "*(18583304.74 + ",
            bessel_y2!(),
            "*(99447.43394 + ",
            bessel_y2!(),
            "*(376.9991397 + ",
            bessel_y2!(),
            ")))))"
        )
    };
}

macro_rules! j1_hankel {
    ($ax:literal) => {
        concat!(
            "(sqrt(2.0 / (3.141592653589793 * (",
            $ax,
            "))) * ((",
            "1.0 + ",
            bessel_yh!($ax),
            "*(0.183105e-2 + ",
            bessel_yh!($ax),
            "*(-3.516396496e-5 + ",
            bessel_yh!($ax),
            "*(2.457520174e-5 - ",
            bessel_yh!($ax),
            "*2.400505341e-7)))",
            ") * cos(",
            bessel_xx!($ax, "2.356194490192345"),
            ") - (",
            bessel_z!($ax),
            ") * (",
            "0.04687499995 + ",
            bessel_yh!($ax),
            "*(-0.2002690873e-3 + ",
            bessel_yh!($ax),
            "*(8.449199096e-5 + ",
            bessel_yh!($ax),
            "*(-8.8228987e-5 + ",
            bessel_yh!($ax),
            "*1.050343160e-6)))",
            ") * sin(",
            bessel_xx!($ax, "2.356194490192345"),
            ")))"
        )
    };
}

macro_rules! wgsl_j0_expr {
    () => {
        concat!(
            "select(select(",
            j0_hankel!("abs(x)"),
            ", ",
            j0_rational!(),
            ", abs(x) < 8.0), 1.0, x == 0.0)"
        )
    };
}

macro_rules! wgsl_j1_expr {
    () => {
        concat!(
            "select((",
            j1_hankel!("abs(x)"),
            ") * sign(x), ",
            j1_rational!(),
            ", abs(x) < 8.0)"
        )
    };
}

macro_rules! c_j0_expr {
    () => {
        concat!(
            "((x) == 0.0 ? 1.0 : ((fabs(x) < 8.0) ? (",
            j0_rational!(),
            ") : (",
            j0_hankel!("fabs(x)"),
            ")))"
        )
    };
}

macro_rules! c_j1_expr {
    () => {
        concat!(
            "((fabs(x) < 8.0) ? (",
            j1_rational!(),
            ") : ((",
            j1_hankel!("fabs(x)"),
            ") * ((x) < 0.0 ? -1.0 : 1.0)))"
        )
    };
}

// J0/J1 stay out of the tables: the WGSL spellings nest `select`, which
// concretizes to `f32` exactly as `SignOp` documents, so they report
// `SUPPORTS_F64 = false` until an f64-capable device proves the spelling
// (the arms are x-expressions, so re-enabling is a flag flip once proven).
// The C spellings use unsuffixed double coefficients with `fabs`/`sin`/
// `cos`/`sqrt` overloads, converting implicitly and serving both precisions.
// CUDA and HIP share one spelling through `c_j0_expr!`/`c_j1_expr!`.
impl UnaryExpr<Wgsl> for J0Op {
    const EXPR: &'static str = wgsl_j0_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for J0Op {
    const EXPR: &'static str = c_j0_expr!();
}
impl UnaryExpr<HipC> for J0Op {
    const EXPR: &'static str = c_j0_expr!();
}
impl UnaryExpr<Wgsl> for J1Op {
    const EXPR: &'static str = wgsl_j1_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for J1Op {
    const EXPR: &'static str = c_j1_expr!();
}
impl UnaryExpr<HipC> for J1Op {
    const EXPR: &'static str = c_j1_expr!();
}

// K0 follows Abramowitz & Stegun 9.8.5 (`0 < x <= 2`: I0 polynomial plus a
// log correction) and 9.8.6 (`x > 2`: scaled exponential series), with the
// same coefficients and nesting as leto's scalar `bessel_k0`. The domain
// guard (`x > 0` and finite, else NaN) needs a NaN literal, which no dialect
// spells: both families use `(x - x) / (x - x)`, NaN for every lane value
// (0/0 and NaN/NaN alike) yet never constant (the lane value is runtime).
// WGSL tests finiteness as `abs(x) <= 3.402823466e+38`, the lgamma-spelling
// magnitude guard: self-subtraction folds to zero under driver fast-math,
// while the magnitude compare is unfoldable and rejects NaN through the
// false comparison. C uses the `isfinite` builtin instead.
macro_rules! k0_t1 {
    () => {
        "(((x) / 3.75) * ((x) / 3.75))"
    };
}

macro_rules! k0_t2 {
    () => {
        "(((x) * 0.5) * ((x) * 0.5))"
    };
}

macro_rules! k0_i0 {
    () => {
        concat!(
            "1.0 + ",
            k0_t1!(),
            "*(3.5156229 + ",
            k0_t1!(),
            "*(3.0899424 + ",
            k0_t1!(),
            "*(1.2067492 + ",
            k0_t1!(),
            "*(0.2659732 + ",
            k0_t1!(),
            "*(0.0360768 + ",
            k0_t1!(),
            "*0.0045813)))))"
        )
    };
}

macro_rules! k0_correction {
    () => {
        concat!(
            "-0.57721566 + ",
            k0_t2!(),
            "*(0.42278420 + ",
            k0_t2!(),
            "*(0.23069756 + ",
            k0_t2!(),
            "*(0.03488590 + ",
            k0_t2!(),
            "*(0.00262698 + ",
            k0_t2!(),
            "*(0.00010750 + ",
            k0_t2!(),
            "*7.4e-6)))))",
        )
    };
}

macro_rules! k0_small {
    () => {
        concat!(
            "(log(((x) * 0.5)) * (-(",
            k0_i0!(),
            ")) + (",
            k0_correction!(),
            "))",
        )
    };
}

macro_rules! k0_t {
    () => {
        "(2.0 / (x))"
    };
}

macro_rules! k0_series {
    () => {
        concat!(
            "1.25331414 + ",
            k0_t!(),
            "*(-0.07832358 + ",
            k0_t!(),
            "*(0.02189568 + ",
            k0_t!(),
            "*(-0.01062446 + ",
            k0_t!(),
            "*(0.00587872 + ",
            k0_t!(),
            "*(-0.00251540 + ",
            k0_t!(),
            "*0.00053208)))))",
        )
    };
}

macro_rules! k0_large {
    () => {
        concat!("((exp(-(x)) / sqrt(x)) * (", k0_series!(), "))",)
    };
}

macro_rules! k0_nan {
    () => {
        "(((x) - (x)) / ((x) - (x)))"
    };
}

macro_rules! wgsl_k0_expr {
    () => {
        concat!(
            "select(",
            k0_nan!(),
            ", select(",
            k0_large!(),
            ", ",
            k0_small!(),
            ", (x) <= 2.0), ((x) > 0.0) && (abs(x) <= 3.402823466e+38))",
        )
    };
}

macro_rules! c_k0_expr {
    () => {
        concat!(
            "(((x) > 0.0 && isfinite(x)) ? (((x) <= 2.0) ? (",
            k0_small!(),
            ") : (",
            k0_large!(),
            ")) : (",
            k0_nan!(),
            "))",
        )
    };
}

// K0 stays out of the tables with J0/J1: the WGSL spelling nests `select`
// and reports `SUPPORTS_F64 = false` per the select precedent, while the C
// spelling converts implicitly and serves both precisions. CUDA and HIP
// share one spelling through `c_k0_expr!`.
impl UnaryExpr<Wgsl> for K0Op {
    const EXPR: &'static str = wgsl_k0_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for K0Op {
    const EXPR: &'static str = c_k0_expr!();
}
impl UnaryExpr<HipC> for K0Op {
    const EXPR: &'static str = c_k0_expr!();
}
