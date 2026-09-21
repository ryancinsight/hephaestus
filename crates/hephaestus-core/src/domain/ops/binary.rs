//! Binary operation markers: arithmetic and scalar-aware comparisons.

use super::{BinaryExpr, TypedBinaryExpr};
use crate::domain::dialect::{CudaC, DialectScalar, HipC, Host, Wgsl};
use eunomia::{NumericElement, RealField};

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

// ── Value functions (ADR 0061) ──────────────────────────────────────────

/// The binary arithmetic operator as a value function, the operator's
/// definition (ADR 0061 Decision 6): generic over every
/// [`NumericElement`] the operator admits. `Add`, `Sub`, `Mul`, and `Div`
/// implement only [`Self::apply`]; [`PowOp`] is real-only and returns `None`
/// from [`Self::apply`], overriding [`Self::apply_real`] instead (the
/// default forwards to [`Self::apply`], which is why `f32`/`f64` `Add`
/// reaches [`Self::apply`] through the [`Self::apply_real`] default — ADR
/// 0061 Verification plan).
pub trait BinaryValue: Copy + Send + Sync + 'static {
    /// `lhs` combined with `rhs`, or `None` when the operator has no
    /// definition over every [`NumericElement`] (only [`PowOp`] today).
    fn apply<T: NumericElement>(lhs: T, rhs: T) -> Option<T>;

    /// `lhs` combined with `rhs` over the reals. Defaults to [`Self::apply`]
    /// (legal: `RealField` implies `NumericElement`); [`PowOp`] overrides it.
    #[inline]
    fn apply_real<T: RealField>(lhs: T, rhs: T) -> Option<T> {
        Self::apply(lhs, rhs)
    }
}

/// Every operator with a value function is a host binary op, mirroring
/// [`crate::CombineValue`]'s [`Host`] blanket.
impl<Op: BinaryValue> BinaryExpr<Host> for Op {
    const EXPR: &'static str = "host";

    fn value<T: NumericElement>(lhs: T, rhs: T) -> Option<T> {
        Op::apply(lhs, rhs)
    }

    fn real_value<T: RealField>(lhs: T, rhs: T) -> Option<T> {
        Op::apply_real(lhs, rhs)
    }
}

/// Integer addition wraps in two's complement (WGSL kernel semantics, ADR
/// 0061 Decision 5); floating-point addition is IEEE-754 (identical to
/// wrapping for floats — `NumericElement::wrapping_add` documents this).
impl BinaryValue for AddOp {
    fn apply<T: NumericElement>(lhs: T, rhs: T) -> Option<T> {
        Some(lhs.wrapping_add(rhs))
    }
}

/// Wraps for integers, IEEE for floats (see [`AddOp`]).
impl BinaryValue for SubOp {
    fn apply<T: NumericElement>(lhs: T, rhs: T) -> Option<T> {
        Some(lhs.wrapping_sub(rhs))
    }
}

/// Wraps for integers, IEEE for floats (see [`AddOp`]).
impl BinaryValue for MulOp {
    fn apply<T: NumericElement>(lhs: T, rhs: T) -> Option<T> {
        Some(lhs.wrapping_mul(rhs))
    }
}

/// IEEE division for floats; for integers, the dividend `lhs` on a zero
/// divisor or `MIN / -1` (ADR 0061 Decision 5, matching WGSL's integer
/// division rule, evidenced against naga's HLSL/MSL writers and constant
/// evaluator).
impl BinaryValue for DivOp {
    fn apply<T: NumericElement>(lhs: T, rhs: T) -> Option<T> {
        Some(lhs.checked_div(rhs).unwrap_or(lhs))
    }
}

/// Real-only: `apply` reports no definition over the full [`NumericElement`]
/// set, so `i32`/`F16` dispatch is the typed host error (ADR 0061
/// Verification plan); `apply_real` carries the actual computation.
impl BinaryValue for PowOp {
    fn apply<T: NumericElement>(_lhs: T, _rhs: T) -> Option<T> {
        None
    }

    fn apply_real<T: RealField>(lhs: T, rhs: T) -> Option<T> {
        Some(lhs.powf(rhs))
    }
}

/// The typed comparison as a value function, producing the type's indicator
/// value — `1` for true, `0` for false — the value-level counterpart of the
/// dialect's mask literal (`select(0.0, 1.0, ...)` in WGSL, `? 1 : 0` in
/// CUDA/HIP). One generic implementation per operator serves every
/// [`NumericElement`] scalar (ADR 0061 Decision 2); NaN comparisons follow
/// `PartialOrd`/`PartialEq`, which already give the IEEE-754 unordered
/// result (every ordered comparison false, `!=` true).
pub trait TypedBinaryValue<T: NumericElement>: Copy + Send + Sync + 'static {
    /// `lhs` compared with `rhs`.
    fn compare(lhs: T, rhs: T) -> T;
}

/// Every operator with a value function is a host typed comparison,
/// mirroring [`crate::CombineValue`]'s [`Host`] blanket.
impl<Op, T> TypedBinaryExpr<Host, T> for Op
where
    Op: TypedBinaryValue<T>,
    T: NumericElement + DialectScalar<Host>,
{
    const EXPR: &'static str = "host";

    fn value(lhs: T, rhs: T) -> Option<T> {
        Some(Op::compare(lhs, rhs))
    }
}

impl<T: NumericElement> TypedBinaryValue<T> for EqOp {
    fn compare(lhs: T, rhs: T) -> T {
        if lhs == rhs { T::ONE } else { T::ZERO }
    }
}

impl<T: NumericElement> TypedBinaryValue<T> for NeOp {
    fn compare(lhs: T, rhs: T) -> T {
        if lhs == rhs { T::ZERO } else { T::ONE }
    }
}

impl<T: NumericElement> TypedBinaryValue<T> for LtOp {
    fn compare(lhs: T, rhs: T) -> T {
        if lhs < rhs { T::ONE } else { T::ZERO }
    }
}

impl<T: NumericElement> TypedBinaryValue<T> for GtOp {
    fn compare(lhs: T, rhs: T) -> T {
        if lhs > rhs { T::ONE } else { T::ZERO }
    }
}

impl<T: NumericElement> TypedBinaryValue<T> for LeOp {
    fn compare(lhs: T, rhs: T) -> T {
        if lhs <= rhs { T::ONE } else { T::ZERO }
    }
}

impl<T: NumericElement> TypedBinaryValue<T> for GeOp {
    fn compare(lhs: T, rhs: T) -> T {
        if lhs >= rhs { T::ONE } else { T::ZERO }
    }
}

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
