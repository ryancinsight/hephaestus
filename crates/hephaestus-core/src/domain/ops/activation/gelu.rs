//! Exact-GELU and GELU-gradient expressions.

use super::super::UnaryExpr;
use super::super::{GeluGradOp, GeluOp};
use crate::domain::dialect::{CudaC, Wgsl};

// `erfc(-x/√2)` through the Numerical Recipes `erfcc` Chebyshev core
// (ADR 0061 Decision 6): the retired Abramowitz-Stegun `erf` above
// `x = -2` lost the `1 + erf` value to cancellation (`41` ULP at
// `x = -1.94`), and the Sun/FreBSD `erfc` rationals below duplicated
// what the Chebyshev core already serves uniformly. `0.5·x·erfc`
// keeps one argument and one composer shared with the grad term.
macro_rules! wgsl_gelu_erfc_expr {
    () => {
        concat!(
            "select((2.0 - ",
            wgsl_erfc_nr_pos!("(abs((x)) * 0.7071067811865476)"),
            "), ",
            wgsl_erfc_nr_pos!("(abs((x)) * 0.7071067811865476)"),
            ", (x) <= 0.0)"
        )
    };
}

// Exact GELU: `0.5·x·erfc(−x/√2)` through the Chebyshev core. The `-inf`
// guard returns `-0.0` (the true limit); the nested selects and
// f32-transcribed coefficients report `SUPPORTS_F64 = false` per the
// `Expm1Op` precedent. The C spelling calls the native `erfc` builtin
// (unsuffixed, so lane promotion keeps f64 served).
impl UnaryExpr<Wgsl> for GeluOp {
    const EXPR: &'static str = concat!(
        "select((0.5 * (x) * ",
        wgsl_gelu_erfc_expr!(),
        "), -0.0, (x) < -3.402823466e+38)"
    );
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for GeluOp {
    const EXPR: &'static str =
        "(x) < -3.402823466e+38f ? -0.0f : 0.5f * x * erfc(-x * 0.7071067811865476f)";
}
// Exact GELU gradient: the `Φ` term reuses `GeluOp`'s Chebyshev composer
// verbatim — zero new coefficients — and adds the Gaussian term once
// (it needs no correction: its decay is the true asymptotic). Both
// infinities guard explicitly. The C spelling likewise calls the native
// `erfc`/`exp` builtins unsuffixed.
impl UnaryExpr<Wgsl> for GeluGradOp {
    const EXPR: &'static str = concat!(
        "select(select((0.5 * ",
        wgsl_gelu_erfc_expr!(),
        " + (x) * exp(-(0.5 * (x) * (x))) * 0.3989422804014327), 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)"
    );
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for GeluGradOp {
    const EXPR: &'static str = "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : 0.5f * erfc(-x * 0.7071067811865476f) + x * exp(-0.5f * x * x) * 0.3989422804014327f)";
}
