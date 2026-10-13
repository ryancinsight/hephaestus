//! Error-function expressions.

use super::super::UnaryExpr;
use super::super::{ErfOp, ErfcOp};
use crate::domain::dialect::{CudaC, Wgsl};

macro_rules! wgsl_erf_expr {
    () => {
        "(sign((x)) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs((x)))) - 1.453152027) * (1.0 / (1.0 + 0.3275911 * abs((x)))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs((x)))) - 0.284496736) * (1.0 / (1.0 + 0.3275911 * abs((x)))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs((x))))) * exp(-((x)) * ((x)))))"
    };
}

macro_rules! wgsl_erfc_expr {
    () => {
        concat!("(1.0 - ", wgsl_erf_expr!(), ")")
    };
}

impl UnaryExpr<Wgsl> for ErfOp {
    const EXPR: &'static str = wgsl_erf_expr!();
}
impl UnaryExpr<CudaC> for ErfOp {
    const EXPR: &'static str = "erf(x)";
}
impl UnaryExpr<Wgsl> for ErfcOp {
    const EXPR: &'static str = wgsl_erfc_expr!();
}
impl UnaryExpr<CudaC> for ErfcOp {
    const EXPR: &'static str = "erfc(x)";
}
