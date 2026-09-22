//! The per-dialect element expression traits every operator marker implements.
//!
//! Four seams, one per operand arity the shader templates bind: unary (`x`),
//! binary and scalar-aware binary (`lhs`, `rhs`), and the associative combine
//! reductions and scans fold with. Each carries the rendered expression as an
//! associated const and, for a dialect that executes operators instead of
//! rendering them, the optional value function ADR 0061 defines.

use crate::domain::dialect::{DialectScalar, KernelDialect};
use eunomia::{NumericElement, RealField};

/// Element expression over the canonical unary operand `x` in dialect `L`.
pub trait UnaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression mapping `x` (e.g. `"exp(-x)"`).
    const EXPR: &'static str;

    /// The unary operator applied to a value, for a dialect that executes
    /// operators instead of rendering them (ADR 0061).
    ///
    /// `None` unless the operator carries [`UnaryValue`](crate::UnaryValue): the
    /// [`Host`](crate::Host) blanket impl overrides this with
    /// [`UnaryValue::apply`](crate::UnaryValue::apply), and an operator implementing this trait for the
    /// host without a value function reports `None`, which the host turns
    /// into a typed error. A seam impl can call this under the
    /// `Op: UnaryExpr<L>` bound it already receives. Real-valued only:
    /// unary and parameterized operators are transcendental functions with no
    /// integer definition (ADR 0061 Decision 2).
    #[must_use]
    fn value<T: RealField>(_x: T) -> Option<T> {
        None
    }
}

/// Element expression over the canonical operands `lhs`, `rhs` in dialect `L`.
pub trait BinaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"`).
    const EXPR: &'static str;

    /// The binary operator applied to two values, admitting every
    /// [`NumericElement`] (ADR 0061 Decision 2 — arithmetic other than
    /// [`PowOp`](crate::PowOp), which is real-only and defines only [`Self::real_value`]).
    ///
    /// `None` unless the operator carries [`BinaryValue`](crate::BinaryValue); see
    /// [`UnaryExpr::value`] for the host-dispatch mechanics this mirrors.
    #[must_use]
    fn value<T: NumericElement>(_lhs: T, _rhs: T) -> Option<T> {
        None
    }

    /// The binary operator applied to two real values.
    ///
    /// Independently `None` by default (never derived from
    /// [`Self::value`]): the [`Host`](crate::Host) blanket overrides both
    /// methods from the two [`BinaryValue`](crate::BinaryValue) methods, so an operator such as
    /// [`PowOp`](crate::PowOp) that defines only the real path leaves [`Self::value`]
    /// reporting `None` for every scalar while this reports the computed
    /// value for `f32`/`f64`.
    #[must_use]
    fn real_value<T: RealField>(_lhs: T, _rhs: T) -> Option<T> {
        None
    }
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

    /// The typed comparison applied to two values (ADR 0061).
    ///
    /// `None` unless the operator carries [`TypedBinaryValue`](crate::TypedBinaryValue); see
    /// [`UnaryExpr::value`] for the host-dispatch mechanics this mirrors.
    #[must_use]
    fn value(_lhs: T, _rhs: T) -> Option<T> {
        None
    }
}

/// Associative combine expression over `lhs`, `rhs` in dialect `L`, used by
/// reductions and scans.
pub trait CombineExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression combining two partial results (e.g. `"max(lhs, rhs)"`).
    const EXPR: &'static str;

    /// The combine applied to two values, for a dialect that executes
    /// operators instead of rendering them (ADR 0061).
    ///
    /// `None` unless the operator carries [`CombineValue`](crate::CombineValue): the
    /// [`Host`](crate::Host) blanket impl overrides this with
    /// [`CombineValue::combine`](crate::CombineValue::combine), and an operator implementing this trait for
    /// the host without a value function reports `None`, which the host turns
    /// into a typed error. A seam impl can call this under the
    /// `Op: CombineExpr<L>` bound it already receives.
    #[must_use]
    fn value<T: NumericElement>(_lhs: T, _rhs: T) -> Option<T> {
        None
    }
}
