//! The runtime-parameter expression seam and its host value function.

use crate::domain::dialect::{Host, KernelDialect};
use eunomia::RealField;

/// Unary expression over `x` and two runtime scalars named `first` and
/// `second` in dialect `L`.
pub trait ParameterizedUnaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression mapping `x`, `first`, and `second` to one output value.
    const EXPR: &'static str;

    /// The operator applied to a value and its two runtime parameters, for a
    /// dialect that executes operators instead of rendering them (ADR 0061).
    ///
    /// `None` unless the operator carries [`ParameterizedUnaryValue`]: the
    /// [`Host`] blanket impl overrides this with
    /// [`ParameterizedUnaryValue::apply`], and an operator implementing this
    /// trait for the host without a value function reports `None`, which the
    /// host turns into a typed error. A seam impl can call this under the
    /// `Op: ParameterizedUnaryExpr<L>` bound it already receives. Real-valued
    /// only, mirroring [`UnaryExpr::value`](crate::UnaryExpr::value): every
    /// parameterized unary operator is a transcendental or comparison
    /// function with no integer definition (ADR 0061 Decision 2).
    #[must_use]
    fn value<T: RealField>(_x: T, _first: T, _second: T) -> Option<T> {
        None
    }
}

/// The parameterized unary operator as a value function, the operator's
/// definition (ADR 0061 Decision 6): real-valued, generic over every
/// [`RealField`] (`f32`/`f64` today), taking `x` and the two runtime scalars
/// `first`/`second` in the same roles [`ParameterizedUnaryExpr::EXPR`]
/// documents per operator.
pub trait ParameterizedUnaryValue: Copy + Send + Sync + 'static {
    /// `x` transformed by the operator under parameters `first` and `second`.
    fn apply<T: RealField>(x: T, first: T, second: T) -> T;
}

/// Every operator with a value function is a host parameterized unary op,
/// mirroring [`crate::UnaryValue`]'s [`Host`] blanket: a downstream
/// `impl ParameterizedUnaryExpr<Host>` without [`ParameterizedUnaryValue`]
/// stays legal and reports `None` from [`ParameterizedUnaryExpr::value`].
impl<Op: ParameterizedUnaryValue> ParameterizedUnaryExpr<Host> for Op {
    const EXPR: &'static str = "host";

    fn value<T: RealField>(x: T, first: T, second: T) -> Option<T> {
        Some(Op::apply(x, first, second))
    }
}
