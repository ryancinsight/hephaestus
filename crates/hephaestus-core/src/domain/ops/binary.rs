//! Binary operation markers: arithmetic and scalar-aware comparisons.

use super::{BinaryExpr, TypedBinaryExpr};
use crate::domain::dialect::{CudaC, HipC, Wgsl};

// ── Binary markers ───────────────────────────────────────────────────────

/// Addition operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AddOp;

/// Subtraction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubOp;

/// Multiplication operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MulOp;

/// Division operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct DivOp;

/// Power operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct PowOp;

/// Element-wise equality comparison marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct EqOp;

/// Element-wise inequality comparison marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeOp;

/// Element-wise less-than comparison marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LtOp;

/// Element-wise greater-than comparison marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GtOp;

/// Element-wise less-than-or-equal comparison marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeOp;

/// Element-wise greater-than-or-equal comparison marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeOp;

impl BinaryExpr<Wgsl> for AddOp {
    const EXPR: &'static str = "lhs + rhs";
}
impl BinaryExpr<CudaC> for AddOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl BinaryExpr<Wgsl> for SubOp {
    const EXPR: &'static str = "lhs - rhs";
}
impl BinaryExpr<CudaC> for SubOp {
    const EXPR: &'static str = "lhs - rhs";
}

impl BinaryExpr<Wgsl> for MulOp {
    const EXPR: &'static str = "lhs * rhs";
}
impl BinaryExpr<CudaC> for MulOp {
    const EXPR: &'static str = "lhs * rhs";
}

impl BinaryExpr<Wgsl> for DivOp {
    const EXPR: &'static str = "lhs / rhs";
}
impl BinaryExpr<CudaC> for DivOp {
    const EXPR: &'static str = "lhs / rhs";
}

impl BinaryExpr<Wgsl> for PowOp {
    const EXPR: &'static str = "pow(lhs, rhs)";
}
impl BinaryExpr<CudaC> for PowOp {
    const EXPR: &'static str = "pow(lhs, rhs)";
}

macro_rules! impl_typed_comparison_exprs {
    (
        $(
            ($op:ty, $wgsl_f32:literal, $wgsl_u32:literal, $wgsl_i32:literal,
                $cuda_f32:literal, $cuda_f64:literal, $cuda_u32:literal, $cuda_i32:literal)
        ),+ $(,)?
    ) => {
        $(
            impl TypedBinaryExpr<Wgsl, f32> for $op {
                const EXPR: &'static str = $wgsl_f32;
            }
            impl TypedBinaryExpr<Wgsl, u32> for $op {
                const EXPR: &'static str = $wgsl_u32;
            }
            impl TypedBinaryExpr<Wgsl, i32> for $op {
                const EXPR: &'static str = $wgsl_i32;
            }
            impl TypedBinaryExpr<CudaC, f32> for $op {
                const EXPR: &'static str = $cuda_f32;
            }
            impl TypedBinaryExpr<CudaC, f64> for $op {
                const EXPR: &'static str = $cuda_f64;
            }
            impl TypedBinaryExpr<CudaC, u32> for $op {
                const EXPR: &'static str = $cuda_u32;
            }
            impl TypedBinaryExpr<CudaC, i32> for $op {
                const EXPR: &'static str = $cuda_i32;
            }
            impl TypedBinaryExpr<HipC, f32> for $op {
                const EXPR: &'static str = $cuda_f32;
            }
            impl TypedBinaryExpr<HipC, u32> for $op {
                const EXPR: &'static str = $cuda_u32;
            }
            impl TypedBinaryExpr<HipC, i32> for $op {
                const EXPR: &'static str = $cuda_i32;
            }
        )+
    };
}

impl_typed_comparison_exprs!(
    (
        EqOp,
        "select(0.0, 1.0, lhs == rhs)",
        "select(0u, 1u, lhs == rhs)",
        "select(0, 1, lhs == rhs)",
        "lhs == rhs ? 1.0f : 0.0f",
        "lhs == rhs ? 1.0 : 0.0",
        "lhs == rhs ? 1u : 0u",
        "lhs == rhs ? 1 : 0"
    ),
    (
        NeOp,
        "select(0.0, 1.0, lhs != rhs)",
        "select(0u, 1u, lhs != rhs)",
        "select(0, 1, lhs != rhs)",
        "lhs != rhs ? 1.0f : 0.0f",
        "lhs != rhs ? 1.0 : 0.0",
        "lhs != rhs ? 1u : 0u",
        "lhs != rhs ? 1 : 0"
    ),
    (
        LtOp,
        "select(0.0, 1.0, lhs < rhs)",
        "select(0u, 1u, lhs < rhs)",
        "select(0, 1, lhs < rhs)",
        "lhs < rhs ? 1.0f : 0.0f",
        "lhs < rhs ? 1.0 : 0.0",
        "lhs < rhs ? 1u : 0u",
        "lhs < rhs ? 1 : 0"
    ),
    (
        GtOp,
        "select(0.0, 1.0, lhs > rhs)",
        "select(0u, 1u, lhs > rhs)",
        "select(0, 1, lhs > rhs)",
        "lhs > rhs ? 1.0f : 0.0f",
        "lhs > rhs ? 1.0 : 0.0",
        "lhs > rhs ? 1u : 0u",
        "lhs > rhs ? 1 : 0"
    ),
    (
        LeOp,
        "select(0.0, 1.0, lhs <= rhs)",
        "select(0u, 1u, lhs <= rhs)",
        "select(0, 1, lhs <= rhs)",
        "lhs <= rhs ? 1.0f : 0.0f",
        "lhs <= rhs ? 1.0 : 0.0",
        "lhs <= rhs ? 1u : 0u",
        "lhs <= rhs ? 1 : 0"
    ),
    (
        GeOp,
        "select(0.0, 1.0, lhs >= rhs)",
        "select(0u, 1u, lhs >= rhs)",
        "select(0, 1, lhs >= rhs)",
        "lhs >= rhs ? 1.0f : 0.0f",
        "lhs >= rhs ? 1.0 : 0.0",
        "lhs >= rhs ? 1u : 0u",
        "lhs >= rhs ? 1 : 0"
    ),
);

macro_rules! impl_hip_binary_exprs {
    ($(($op:ty, $expr:literal)),+ $(,)?) => {
        $(
            impl BinaryExpr<HipC> for $op {
                const EXPR: &'static str = $expr;
            }
        )+
    };
}

impl_hip_binary_exprs!(
    (AddOp, "lhs + rhs"),
    (SubOp, "lhs - rhs"),
    (MulOp, "lhs * rhs"),
    (DivOp, "lhs / rhs"),
    (PowOp, "pow(lhs, rhs)"),
);
