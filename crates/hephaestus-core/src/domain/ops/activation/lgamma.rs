//! Log-gamma expression.

use super::super::LgammaOp;
use super::super::UnaryExpr;
use crate::domain::dialect::{CudaC, Wgsl};

macro_rules! wgsl_lgamma_positive_expr {
    ($value:literal) => {
        concat!(
            "(0.9189385332046727 + ((",
            $value,
            ") - 0.5) * log((",
            $value,
            ") + 6.5) - ((",
            $value,
            ") + 6.5) + log(0.9999999999998099 + 676.5203681218851 / (",
            $value,
            ") - 1259.1392167224028 / ((",
            $value,
            ") + 1.0) + 771.3234287776531 / ((",
            $value,
            ") + 2.0) - 176.6150291621406 / ((",
            $value,
            ") + 3.0) + 12.5073432786869 / ((",
            $value,
            ") + 4.0) - 0.13857109526572 / ((",
            $value,
            ") + 5.0) + 9.98436957801957e-6 / ((",
            $value,
            ") + 6.0) + 1.50563273514931e-7 / ((",
            $value,
            ") + 7.0)))"
        )
    };
}

macro_rules! wgsl_lgamma_expr {
    () => {
        concat!(
            "select(select(select(",
            "(1.1447298858494002 - log(abs(sin(3.141592653589793 * x))) - ",
            wgsl_lgamma_positive_expr!("max(abs(1.0 - x), 0.5)"),
            "), ",
            wgsl_lgamma_positive_expr!("max(abs(x), 0.5)"),
            ", x >= 0.5), 1.0 / abs(x - trunc(x)), ((x <= 0.0) && (x == trunc(x)))), abs(x), abs(x) > 3.402823466e+38)"
        )
    };
}

impl UnaryExpr<Wgsl> for LgammaOp {
    const EXPR: &'static str = wgsl_lgamma_expr!();
}
impl UnaryExpr<CudaC> for LgammaOp {
    const EXPR: &'static str = "lgamma(x)";
}
