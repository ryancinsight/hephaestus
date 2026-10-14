//! Expression and identity tests for the operation markers.

use super::*;
use crate::domain::dialect::{CudaC, HipC, Host, Wgsl};
use eunomia::RealField;

mod double_double;

use double_double::DoubleDouble;

#[path = "tests/binary_dispatch/mod.rs"]
mod binary_dispatch;
#[path = "tests/fixtures.rs"]
mod fixtures;
#[path = "tests/gelu.rs"]
mod gelu;
#[path = "tests/hardswish.rs"]
mod hardswish;
#[path = "tests/mish.rs"]
mod mish;
#[path = "tests/silu.rs"]
mod silu;
#[path = "tests/special_values.rs"]
mod special_values;
#[path = "tests/tanh.rs"]
mod tanh;
pub(crate) use fixtures::*;

#[test]
fn wgsl_erfc_family_expressions_balance() {
    // The composed Chebyshev strings are long enough to mis-parenthesize by
    // hand; naga rejects the imbalance at dispatch, but this pins balance
    // in under a millisecond with no device.
    for expr in [
        <ErfcOp as UnaryExpr<Wgsl>>::EXPR,
        <GeluOp as UnaryExpr<Wgsl>>::EXPR,
        <GeluGradOp as UnaryExpr<Wgsl>>::EXPR,
    ] {
        let mut depth = 0i32;
        for byte in expr.bytes() {
            if byte == b'(' {
                depth += 1;
            }
            if byte == b')' {
                depth -= 1;
            }
            assert!(depth >= 0, "unbalanced parens in: {expr}");
        }
        assert_eq!(depth, 0, "unbalanced parens in: {expr}");
    }
}

#[test]
fn gated_family_expressions_balance() {
    // The self-gated assemblies splice shared composers up to seven times
    // (mish-grad's softplus); a dropped paren anywhere mis-nests the whole
    // rendering. Naga rejects it at dispatch, but this pins balance in under
    // a millisecond with no device. The C strings serve both C dialects, so
    // one spelling per op suffices there.
    for expr in [
        <SiluOp as UnaryExpr<Wgsl>>::EXPR,
        <SiluGradOp as UnaryExpr<Wgsl>>::EXPR,
        <MishOp as UnaryExpr<Wgsl>>::EXPR,
        <MishGradOp as UnaryExpr<Wgsl>>::EXPR,
        <GeluTanhOp as UnaryExpr<Wgsl>>::EXPR,
        <GeluTanhGradOp as UnaryExpr<Wgsl>>::EXPR,
        <SiluOp as UnaryExpr<CudaC>>::EXPR,
        <SiluGradOp as UnaryExpr<CudaC>>::EXPR,
        <MishOp as UnaryExpr<CudaC>>::EXPR,
        <MishGradOp as UnaryExpr<CudaC>>::EXPR,
        <GeluTanhOp as UnaryExpr<CudaC>>::EXPR,
        <GeluTanhGradOp as UnaryExpr<CudaC>>::EXPR,
    ] {
        let mut depth = 0i32;
        for byte in expr.bytes() {
            if byte == b'(' {
                depth += 1;
            }
            if byte == b')' {
                depth -= 1;
            }
            assert!(depth >= 0, "unbalanced parens in: {expr}");
        }
        assert_eq!(depth, 0, "unbalanced parens in: {expr}");
    }
}
