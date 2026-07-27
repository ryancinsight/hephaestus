//! Zero-sized operation markers with per-dialect shader expressions.
//!
//! One op vocabulary for every backend: each marker is a ZST whose dialect
//! expression is an associated const on a
//! [`KernelDialect`](crate::KernelDialect)-parameterized
//! trait, so backend shader templates substitute `Op::EXPR` for their own
//! dialect and dispatch stays fully monomorphized. Consumers add fused ops
//! without touching this crate by implementing the expression trait for
//! their own ZST in the dialects they target — a kernel authored for one
//! dialect does not compile on a backend of another dialect.
//!
//! Canonical operand names (backend templates must bind these locals):
//! - unary expressions read `x`;
//! - binary and combine expressions read `lhs` and `rhs`.

use super::dialect::{CudaC, DialectScalar, HipC, KernelDialect, Wgsl};
use bytemuck::Pod;

/// Element expression over the canonical unary operand `x` in dialect `L`.
pub trait UnaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression mapping `x` (e.g. `"exp(-x)"`).
    const EXPR: &'static str;
}

/// Element expression over the canonical operands `lhs`, `rhs` in dialect `L`.
pub trait BinaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"`).
    const EXPR: &'static str;
}

/// Scalar-aware binary expression over the canonical operands `lhs`, `rhs`.
///
/// This seam is required when the result expression depends on the scalar
/// representation. Comparisons are the current example: WGSL and CUDA/HIP
/// require different zero/one literal tokens for floating-point and integer
/// masks. Arithmetic operations should use [`BinaryExpr`] when one expression
/// is valid for every scalar supported by the operation.
pub trait TypedBinaryExpr<L: KernelDialect, T: DialectScalar<L>>:
    Copy + Send + Sync + 'static
{
    /// Expression combining `lhs` and `rhs` for scalar `T` in dialect `L`.
    const EXPR: &'static str;
}

/// Associative combine expression over `lhs`, `rhs` in dialect `L`, used by
/// reductions and scans.
pub trait CombineExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression combining two partial results (e.g. `"max(lhs, rhs)"`).
    const EXPR: &'static str;
}

/// Host-side identity element of op `Op` for this scalar (dialect-free).
pub trait OpIdentity<Op>: Pod {
    /// The identity value (e.g. `0` for sum, `T::MAX` for min).
    const IDENTITY: Self;
}

/// Shader literal token of op `Op`'s identity for this scalar in dialect `L`.
pub trait IdentityToken<Op, L: KernelDialect>: DialectScalar<L> {
    /// The dialect literal (e.g. `"0.0"` in WGSL, `"0.0f"` in CUDA C++).
    const TOKEN: &'static str;
}

// ── Unary markers ────────────────────────────────────────────────────────

/// Exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpOp;

/// Fused negated exponential `exp(-x)` marker (e.g. Beer–Lambert
/// transmission).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpNegOp;

/// Natural logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LnOp;

/// Sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinOp;

/// Cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosOp;

/// Square-root operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SqrtOp;

/// Absolute value operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AbsOp;

/// Negation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct NegOp;

/// Reciprocal operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecipOp;

/// Identity/copy operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityOp;

/// Tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanOp;

/// Arcsine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsinOp;

/// Arccosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcosOp;

/// Arctangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtanOp;

/// Hyperbolic sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinhOp;

/// Hyperbolic cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoshOp;

/// Base-two logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log2Op;

/// Base-ten logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log10Op;

/// Base-two exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exp2Op;

/// Inverse hyperbolic tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtanhOp;

/// Inverse hyperbolic sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsinhOp;

/// Inverse hyperbolic cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcoshOp;

/// Exponential-minus-one operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Expm1Op;

/// Logarithm-of-one-plus operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log1pOp;

/// Sign operation marker, returning `-1`, `0`, or `1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignOp;

/// Floor operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct FloorOp;

/// Ceiling operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CeilOp;

/// Round-to-nearest-even operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundOp;

/// Truncation-toward-zero operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TruncOp;

/// Gauss error function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErfOp;

/// Complementary Gauss error function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErfcOp;

/// Rectified linear unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluOp;

/// Rectified linear unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluGradOp;

/// Logistic sigmoid operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidOp;

/// Logistic sigmoid gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidGradOp;

/// Hyperbolic tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhOp;

/// Hyperbolic tangent gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhGradOp;

/// Tanh-approximated Gaussian Error Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhOp;

/// Tanh-approximated Gaussian Error Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhGradOp;

/// Sigmoid Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluOp;

/// Sigmoid Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluGradOp;

/// Softplus operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusOp;

/// Softplus gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusGradOp;

impl UnaryExpr<Wgsl> for ExpOp {
    const EXPR: &'static str = "exp(x)";
}
impl UnaryExpr<CudaC> for ExpOp {
    const EXPR: &'static str = "exp(x)";
}

impl UnaryExpr<Wgsl> for ExpNegOp {
    const EXPR: &'static str = "exp(-x)";
}
impl UnaryExpr<CudaC> for ExpNegOp {
    const EXPR: &'static str = "exp(-x)";
}

impl UnaryExpr<Wgsl> for LnOp {
    const EXPR: &'static str = "log(x)";
}
impl UnaryExpr<CudaC> for LnOp {
    const EXPR: &'static str = "log(x)";
}

impl UnaryExpr<Wgsl> for SinOp {
    const EXPR: &'static str = "sin(x)";
}
impl UnaryExpr<CudaC> for SinOp {
    const EXPR: &'static str = "sin(x)";
}

impl UnaryExpr<Wgsl> for CosOp {
    const EXPR: &'static str = "cos(x)";
}
impl UnaryExpr<CudaC> for CosOp {
    const EXPR: &'static str = "cos(x)";
}

impl UnaryExpr<Wgsl> for SqrtOp {
    const EXPR: &'static str = "sqrt(x)";
}
impl UnaryExpr<CudaC> for SqrtOp {
    const EXPR: &'static str = "sqrt(x)";
}

impl UnaryExpr<Wgsl> for AbsOp {
    const EXPR: &'static str = "abs(x)";
}
impl UnaryExpr<CudaC> for AbsOp {
    const EXPR: &'static str = "abs(x)";
}

impl UnaryExpr<Wgsl> for NegOp {
    const EXPR: &'static str = "-x";
}
impl UnaryExpr<CudaC> for NegOp {
    const EXPR: &'static str = "-x";
}

impl UnaryExpr<Wgsl> for RecipOp {
    const EXPR: &'static str = "1.0 / x";
}
impl UnaryExpr<CudaC> for RecipOp {
    const EXPR: &'static str = "1.0 / x";
}

impl UnaryExpr<Wgsl> for IdentityOp {
    const EXPR: &'static str = "x";
}
impl UnaryExpr<CudaC> for IdentityOp {
    const EXPR: &'static str = "x";
}

macro_rules! impl_math_unary_exprs {
    ($(($op:ty, $wgsl:literal, $cuda:literal)),+ $(,)?) => {
        $(
            impl UnaryExpr<Wgsl> for $op {
                const EXPR: &'static str = $wgsl;
            }
            impl UnaryExpr<CudaC> for $op {
                const EXPR: &'static str = $cuda;
            }
        )+
    };
}

impl_math_unary_exprs!(
    (TanOp, "tan(x)", "tan(x)"),
    (AsinOp, "asin(x)", "asin(x)"),
    (AcosOp, "acos(x)", "acos(x)"),
    (AtanOp, "atan(x)", "atan(x)"),
    (SinhOp, "sinh(x)", "sinh(x)"),
    (CoshOp, "cosh(x)", "cosh(x)"),
    (Log2Op, "log2(x)", "log2(x)"),
    (
        Log10Op,
        "log(x) * 0.43429448190325182f",
        "log(x) * 0.43429448190325182f"
    ),
    (Exp2Op, "exp2(x)", "exp2(x)"),
    (AtanhOp, "atanh(x)", "atanh(x)"),
    (AsinhOp, "asinh(x)", "asinh(x)"),
    (AcoshOp, "acosh(x)", "acosh(x)"),
    (Expm1Op, "(exp(x) - 1.0)", "(exp(x) - 1.0f)"),
    (Log1pOp, "log(1.0 + (x))", "log(1.0f + (x))"),
    (
        SignOp,
        "select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0)",
        "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)"
    ),
    (FloorOp, "floor(x)", "floor(x)"),
    (CeilOp, "ceil(x)", "ceil(x)"),
    (RoundOp, "round(x)", "rint(x)"),
    (TruncOp, "trunc(x)", "trunc(x)"),
);

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

macro_rules! impl_activation_unary_exprs {
    ($(($op:ty, $wgsl:literal, $cuda:literal)),+ $(,)?) => {
        $(
            impl UnaryExpr<Wgsl> for $op {
                const EXPR: &'static str = $wgsl;
            }
            impl UnaryExpr<CudaC> for $op {
                const EXPR: &'static str = $cuda;
            }
        )+
    };
}

impl_activation_unary_exprs!(
    (ReluOp, "max(x, 0.0)", "max(x, 0.0f)"),
    (
        ReluGradOp,
        "select(0.0, 1.0, x > 0.0)",
        "x > 0.0f ? 1.0f : 0.0f"
    ),
    (
        SigmoidOp,
        "1.0 / (1.0 + exp(-x))",
        "1.0f / (1.0f + exp(-x))"
    ),
    (SigmoidGradOp, "x * (1.0 - x)", "x * (1.0f - x)"),
    (TanhOp, "tanh(x)", "tanh(x)"),
    (TanhGradOp, "1.0 - x * x", "1.0f - x * x"),
    (
        GeluTanhOp,
        "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
    ),
    (
        GeluTanhGradOp,
        "0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))) + 0.5 * x * (1.0 - tanh(0.7978845608 * (x + 0.044715 * x * x * x)) * tanh(0.7978845608 * (x + 0.044715 * x * x * x))) * 0.7978845608 * (1.0 + 0.134145 * x * x)",
        "0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + 0.5f * x * (1.0f - tanh(0.7978845608f * (x + 0.044715f * x * x * x)) * tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * 0.7978845608f * (1.0f + 0.134145f * x * x)"
    ),
    (SiluOp, "x / (1.0 + exp(-x))", "x / (1.0f + exp(-x))"),
    (
        SiluGradOp,
        "(1.0 / (1.0 + exp(-x))) * (1.0 + x * (1.0 - (1.0 / (1.0 + exp(-x)))))",
        "(1.0f / (1.0f + exp(-x))) * (1.0f + x * (1.0f - (1.0f / (1.0f + exp(-x)))))"
    ),
    (SoftplusOp, "log(1.0 + exp(x))", "log(1.0f + exp(x))"),
    (
        SoftplusGradOp,
        "1.0 / (1.0 + exp(-x))",
        "1.0f / (1.0f + exp(-x))"
    ),
);

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
                $cuda_f32:literal, $cuda_u32:literal, $cuda_i32:literal)
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
        "lhs == rhs ? 1u : 0u",
        "lhs == rhs ? 1 : 0"
    ),
    (
        NeOp,
        "select(0.0, 1.0, lhs != rhs)",
        "select(0u, 1u, lhs != rhs)",
        "select(0, 1, lhs != rhs)",
        "lhs != rhs ? 1.0f : 0.0f",
        "lhs != rhs ? 1u : 0u",
        "lhs != rhs ? 1 : 0"
    ),
    (
        LtOp,
        "select(0.0, 1.0, lhs < rhs)",
        "select(0u, 1u, lhs < rhs)",
        "select(0, 1, lhs < rhs)",
        "lhs < rhs ? 1.0f : 0.0f",
        "lhs < rhs ? 1u : 0u",
        "lhs < rhs ? 1 : 0"
    ),
    (
        GtOp,
        "select(0.0, 1.0, lhs > rhs)",
        "select(0u, 1u, lhs > rhs)",
        "select(0, 1, lhs > rhs)",
        "lhs > rhs ? 1.0f : 0.0f",
        "lhs > rhs ? 1u : 0u",
        "lhs > rhs ? 1 : 0"
    ),
    (
        LeOp,
        "select(0.0, 1.0, lhs <= rhs)",
        "select(0u, 1u, lhs <= rhs)",
        "select(0, 1, lhs <= rhs)",
        "lhs <= rhs ? 1.0f : 0.0f",
        "lhs <= rhs ? 1u : 0u",
        "lhs <= rhs ? 1 : 0"
    ),
    (
        GeOp,
        "select(0.0, 1.0, lhs >= rhs)",
        "select(0u, 1u, lhs >= rhs)",
        "select(0, 1, lhs >= rhs)",
        "lhs >= rhs ? 1.0f : 0.0f",
        "lhs >= rhs ? 1u : 0u",
        "lhs >= rhs ? 1 : 0"
    ),
);

// ── Reduction markers ────────────────────────────────────────────────────

/// Sum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumOp;

/// Product-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProdOp;

/// Minimum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinOp;

/// Maximum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxOp;

impl CombineExpr<Wgsl> for SumOp {
    const EXPR: &'static str = "lhs + rhs";
}
impl CombineExpr<CudaC> for SumOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl CombineExpr<Wgsl> for ProdOp {
    const EXPR: &'static str = "lhs * rhs";
}
impl CombineExpr<CudaC> for ProdOp {
    const EXPR: &'static str = "lhs * rhs";
}

impl CombineExpr<Wgsl> for MinOp {
    const EXPR: &'static str = "min(lhs, rhs)";
}
impl CombineExpr<CudaC> for MinOp {
    const EXPR: &'static str = "min(lhs, rhs)";
}

impl CombineExpr<Wgsl> for MaxOp {
    const EXPR: &'static str = "max(lhs, rhs)";
}
impl CombineExpr<CudaC> for MaxOp {
    const EXPR: &'static str = "max(lhs, rhs)";
}

// ── Scan markers ─────────────────────────────────────────────────────────

/// Cumulative-sum scan operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumSumOp;

/// Cumulative-product scan operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumProdOp;

impl CombineExpr<Wgsl> for CumSumOp {
    const EXPR: &'static str = "lhs + rhs";
}
impl CombineExpr<CudaC> for CumSumOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl CombineExpr<Wgsl> for CumProdOp {
    const EXPR: &'static str = "lhs * rhs";
}
impl CombineExpr<CudaC> for CumProdOp {
    const EXPR: &'static str = "lhs * rhs";
}

macro_rules! impl_hip_unary_exprs {
    ($(($op:ty, $expr:literal)),+ $(,)?) => {
        $(
            impl UnaryExpr<HipC> for $op {
                const EXPR: &'static str = $expr;
            }
        )+
    };
}

impl_hip_unary_exprs!(
    (ExpOp, "exp(x)"),
    (ExpNegOp, "exp(-x)"),
    (LnOp, "log(x)"),
    (SinOp, "sin(x)"),
    (CosOp, "cos(x)"),
    (SqrtOp, "sqrt(x)"),
    (AbsOp, "abs(x)"),
    (NegOp, "-x"),
    (RecipOp, "1.0 / x"),
    (IdentityOp, "x"),
    (TanOp, "tan(x)"),
    (AsinOp, "asin(x)"),
    (AcosOp, "acos(x)"),
    (AtanOp, "atan(x)"),
    (SinhOp, "sinh(x)"),
    (CoshOp, "cosh(x)"),
    (Log2Op, "log2(x)"),
    (Log10Op, "log(x) * 0.43429448190325182f"),
    (Exp2Op, "exp2(x)"),
    (AtanhOp, "atanh(x)"),
    (AsinhOp, "asinh(x)"),
    (AcoshOp, "acosh(x)"),
    (Expm1Op, "(exp(x) - 1.0f)"),
    (Log1pOp, "log(1.0f + (x))"),
    (SignOp, "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)"),
    (FloorOp, "floor(x)"),
    (CeilOp, "ceil(x)"),
    (RoundOp, "rint(x)"),
    (TruncOp, "trunc(x)"),
    (ErfOp, "erf(x)"),
    (ErfcOp, "erfc(x)"),
    (ReluOp, "max(x, 0.0f)"),
    (ReluGradOp, "x > 0.0f ? 1.0f : 0.0f"),
    (SigmoidOp, "1.0f / (1.0f + exp(-x))"),
    (SigmoidGradOp, "x * (1.0f - x)"),
    (TanhOp, "tanh(x)"),
    (TanhGradOp, "1.0f - x * x"),
    (
        GeluTanhOp,
        "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
    ),
    (
        GeluTanhGradOp,
        "0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + 0.5f * x * (1.0f - tanh(0.7978845608f * (x + 0.044715f * x * x * x)) * tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * 0.7978845608f * (1.0f + 0.134145f * x * x)"
    ),
    (SiluOp, "x / (1.0f + exp(-x))"),
    (
        SiluGradOp,
        "(1.0f / (1.0f + exp(-x))) * (1.0f + x * (1.0f - (1.0f / (1.0f + exp(-x)))))"
    ),
    (SoftplusOp, "log(1.0f + exp(x))"),
    (SoftplusGradOp, "1.0f / (1.0f + exp(-x))"),
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

macro_rules! impl_hip_combine_exprs {
    ($(($op:ty, $expr:literal)),+ $(,)?) => {
        $(
            impl CombineExpr<HipC> for $op {
                const EXPR: &'static str = $expr;
            }
        )+
    };
}

impl_hip_combine_exprs!(
    (SumOp, "lhs + rhs"),
    (ProdOp, "lhs * rhs"),
    (MinOp, "min(lhs, rhs)"),
    (MaxOp, "max(lhs, rhs)"),
    (CumSumOp, "lhs + rhs"),
    (CumProdOp, "lhs * rhs"),
);

// ── Identities ───────────────────────────────────────────────────────────
// Host values are dialect-free; literal tokens differ per dialect (WGSL has
// no `f` suffix, CUDA C++ float literals carry one so arithmetic stays in
// `float` rather than promoting to `double`).

impl OpIdentity<SumOp> for f32 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<SumOp> for u32 {
    const IDENTITY: Self = 0;
}
impl OpIdentity<SumOp> for i32 {
    const IDENTITY: Self = 0;
}

impl OpIdentity<ProdOp> for f32 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<ProdOp> for u32 {
    const IDENTITY: Self = 1;
}
impl OpIdentity<ProdOp> for i32 {
    const IDENTITY: Self = 1;
}

impl OpIdentity<MinOp> for f32 {
    const IDENTITY: Self = f32::MAX;
}
impl OpIdentity<MinOp> for u32 {
    const IDENTITY: Self = u32::MAX;
}
impl OpIdentity<MinOp> for i32 {
    const IDENTITY: Self = i32::MAX;
}

impl OpIdentity<MaxOp> for f32 {
    const IDENTITY: Self = f32::MIN;
}
impl OpIdentity<MaxOp> for u32 {
    const IDENTITY: Self = u32::MIN;
}
impl OpIdentity<MaxOp> for i32 {
    const IDENTITY: Self = i32::MIN;
}

impl OpIdentity<CumSumOp> for f32 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<CumSumOp> for u32 {
    const IDENTITY: Self = 0;
}
impl OpIdentity<CumSumOp> for i32 {
    const IDENTITY: Self = 0;
}

impl OpIdentity<CumProdOp> for f32 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<CumProdOp> for u32 {
    const IDENTITY: Self = 1;
}
impl OpIdentity<CumProdOp> for i32 {
    const IDENTITY: Self = 1;
}

impl IdentityToken<SumOp, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<SumOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<SumOp, Wgsl> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<SumOp, CudaC> for f32 {
    const TOKEN: &'static str = "0.0f";
}
impl IdentityToken<SumOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<SumOp, CudaC> for i32 {
    const TOKEN: &'static str = "0";
}

impl IdentityToken<ProdOp, Wgsl> for f32 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<ProdOp, Wgsl> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<ProdOp, Wgsl> for i32 {
    const TOKEN: &'static str = "1";
}
impl IdentityToken<ProdOp, CudaC> for f32 {
    const TOKEN: &'static str = "1.0f";
}
impl IdentityToken<ProdOp, CudaC> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<ProdOp, CudaC> for i32 {
    const TOKEN: &'static str = "1";
}

impl IdentityToken<MinOp, Wgsl> for f32 {
    const TOKEN: &'static str = "3.402823466e+38";
}
impl IdentityToken<MinOp, Wgsl> for u32 {
    const TOKEN: &'static str = "4294967295u";
}
impl IdentityToken<MinOp, Wgsl> for i32 {
    const TOKEN: &'static str = "2147483647";
}
impl IdentityToken<MinOp, CudaC> for f32 {
    const TOKEN: &'static str = "3.402823466e+38f";
}
impl IdentityToken<MinOp, CudaC> for u32 {
    const TOKEN: &'static str = "4294967295u";
}
impl IdentityToken<MinOp, CudaC> for i32 {
    const TOKEN: &'static str = "2147483647";
}

impl IdentityToken<MaxOp, Wgsl> for f32 {
    const TOKEN: &'static str = "-3.402823466e+38";
}
impl IdentityToken<MaxOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<MaxOp, Wgsl> for i32 {
    const TOKEN: &'static str = "-2147483648";
}
impl IdentityToken<MaxOp, CudaC> for f32 {
    const TOKEN: &'static str = "-3.402823466e+38f";
}
impl IdentityToken<MaxOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<MaxOp, CudaC> for i32 {
    const TOKEN: &'static str = "-2147483648";
}

impl IdentityToken<CumSumOp, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<CumSumOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<CumSumOp, Wgsl> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<CumSumOp, CudaC> for f32 {
    const TOKEN: &'static str = "0.0f";
}
impl IdentityToken<CumSumOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<CumSumOp, CudaC> for i32 {
    const TOKEN: &'static str = "0";
}

impl IdentityToken<CumProdOp, Wgsl> for f32 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<CumProdOp, Wgsl> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<CumProdOp, Wgsl> for i32 {
    const TOKEN: &'static str = "1";
}
impl IdentityToken<CumProdOp, CudaC> for f32 {
    const TOKEN: &'static str = "1.0f";
}
impl IdentityToken<CumProdOp, CudaC> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<CumProdOp, CudaC> for i32 {
    const TOKEN: &'static str = "1";
}

macro_rules! impl_hip_identity_tokens {
    ($(($op:ty, $f32_token:literal, $u32_token:literal, $i32_token:literal)),+ $(,)?) => {
        $(
            impl IdentityToken<$op, HipC> for f32 {
                const TOKEN: &'static str = $f32_token;
            }
            impl IdentityToken<$op, HipC> for u32 {
                const TOKEN: &'static str = $u32_token;
            }
            impl IdentityToken<$op, HipC> for i32 {
                const TOKEN: &'static str = $i32_token;
            }
        )+
    };
}

impl_hip_identity_tokens!(
    (SumOp, "0.0f", "0u", "0"),
    (ProdOp, "1.0f", "1u", "1"),
    (MinOp, "3.402823466e+38f", "4294967295u", "2147483647"),
    (MaxOp, "-3.402823466e+38f", "0u", "-2147483648"),
    (CumSumOp, "0.0f", "0u", "0"),
    (CumProdOp, "1.0f", "1u", "1"),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_and_identity_agree_per_dialect() {
        assert_eq!(<SumOp as CombineExpr<Wgsl>>::EXPR, "lhs + rhs");
        assert_eq!(<SumOp as CombineExpr<CudaC>>::EXPR, "lhs + rhs");
        assert_eq!(<SumOp as CombineExpr<HipC>>::EXPR, "lhs + rhs");
        assert_eq!(<ProdOp as CombineExpr<HipC>>::EXPR, "lhs * rhs");
        assert_eq!(<AddOp as BinaryExpr<HipC>>::EXPR, "lhs + rhs");
        assert_eq!(<NegOp as UnaryExpr<HipC>>::EXPR, "-x");
        assert_eq!(
            <GeluTanhOp as UnaryExpr<Wgsl>>::EXPR,
            "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))"
        );
        assert!(<SiluGradOp as UnaryExpr<CudaC>>::EXPR.contains("exp(-x)"));
        assert!(<SoftplusOp as UnaryExpr<HipC>>::EXPR.contains("exp(x)"));
        assert_eq!(
            <Log10Op as UnaryExpr<Wgsl>>::EXPR,
            "log(x) * 0.43429448190325182f"
        );
        assert_eq!(<Expm1Op as UnaryExpr<CudaC>>::EXPR, "(exp(x) - 1.0f)");
        assert_eq!(<RoundOp as UnaryExpr<HipC>>::EXPR, "rint(x)");
        assert_eq!(<ErfOp as UnaryExpr<CudaC>>::EXPR, "erf(x)");
        assert_eq!(<ErfcOp as UnaryExpr<HipC>>::EXPR, "erfc(x)");
        assert!(<ErfOp as UnaryExpr<Wgsl>>::EXPR.contains("1.061405429"));
        assert!(<ErfcOp as UnaryExpr<Wgsl>>::EXPR.starts_with("(1.0 - "));
        assert_eq!(
            <SignOp as UnaryExpr<Wgsl>>::EXPR,
            "select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0)"
        );
        assert_eq!(<f32 as IdentityToken<SumOp, Wgsl>>::TOKEN, "0.0");
        assert_eq!(<f32 as IdentityToken<SumOp, CudaC>>::TOKEN, "0.0f");
        assert_eq!(<f32 as IdentityToken<SumOp, HipC>>::TOKEN, "0.0f");
        assert_eq!(<f32 as OpIdentity<MinOp>>::IDENTITY, f32::MAX);
        assert_eq!(<f32 as OpIdentity<ProdOp>>::IDENTITY, 1.0);
        assert_eq!(<u32 as OpIdentity<MaxOp>>::IDENTITY, u32::MIN);
    }

    #[test]
    fn comparisons_use_scalar_correct_mask_literals() {
        assert_eq!(
            <EqOp as TypedBinaryExpr<Wgsl, f32>>::EXPR,
            "select(0.0, 1.0, lhs == rhs)"
        );
        assert_eq!(
            <EqOp as TypedBinaryExpr<Wgsl, u32>>::EXPR,
            "select(0u, 1u, lhs == rhs)"
        );
        assert_eq!(
            <EqOp as TypedBinaryExpr<Wgsl, i32>>::EXPR,
            "select(0, 1, lhs == rhs)"
        );
        assert_eq!(
            <GeOp as TypedBinaryExpr<CudaC, f32>>::EXPR,
            "lhs >= rhs ? 1.0f : 0.0f"
        );
        assert_eq!(
            <GeOp as TypedBinaryExpr<CudaC, u32>>::EXPR,
            "lhs >= rhs ? 1u : 0u"
        );
        assert_eq!(
            <GeOp as TypedBinaryExpr<HipC, i32>>::EXPR,
            "lhs >= rhs ? 1 : 0"
        );
    }

    #[test]
    fn consumer_defined_op_composes_with_the_vocabulary() {
        // A consumer-side fused op: implement the expression trait for a
        // local ZST in the targeted dialect — no substrate changes needed.
        #[derive(Clone, Copy, Debug, Default)]
        struct AffineClampOp;
        impl UnaryExpr<Wgsl> for AffineClampOp {
            const EXPR: &'static str = "clamp(x * 2.0 + 1.0, 0.0, 10.0)";
        }
        fn expr_of<Op: UnaryExpr<Wgsl>>() -> &'static str {
            Op::EXPR
        }
        assert_eq!(
            expr_of::<AffineClampOp>(),
            "clamp(x * 2.0 + 1.0, 0.0, 10.0)"
        );
    }
}
