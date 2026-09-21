//! Value functions for the elementary unary markers (ADR 0061).
//!
//! Covers the operators whose dialect expressions live in `unary.rs`'s own
//! macro block: direct transcendental, rounding, and sign operations with no
//! cancellation hazard beyond what eunomia's `FloatElement` already resolves.
//! The activation family (sigmoid/tanh/softplus/gelu/silu/mish/elu and their
//! gradients, plus erf/erfc/lgamma) lives in `activation_value.rs`, where the
//! formulations avoid the tail cancellation ADR 0061 Decision 6 documents.

use super::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, Exp2Op,
    ExpNegOp, ExpOp, Expm1Op, FloorOp, IdentityOp, LnOp, Log1pOp, Log2Op, Log10Op, NegOp, RecipOp,
    RoundOp, SignOp, SinOp, SinhOp, SqrtOp, TanOp, TruncOp, UnaryExpr,
};
use crate::domain::dialect::Host;
use eunomia::{NumericElement, RealField};

/// The unary operator as a value function, the operator's definition
/// (ADR 0061 Decision 6): real-valued, generic over every
/// [`RealField`] (`f32`/`f64` today).
pub trait UnaryValue: Copy + Send + Sync + 'static {
    /// `x` transformed by the operator.
    fn apply<T: RealField>(x: T) -> T;
}

/// Every operator with a value function is a host unary op, mirroring
/// [`crate::CombineValue`]'s [`Host`] blanket: a downstream
/// `impl UnaryExpr<Host>` without [`UnaryValue`] stays legal and reports
/// `None` from [`UnaryExpr::value`].
impl<Op: UnaryValue> UnaryExpr<Host> for Op {
    const EXPR: &'static str = "host";

    fn value<T: RealField>(x: T) -> Option<T> {
        Some(Op::apply(x))
    }
}

macro_rules! impl_direct_unary_value {
    ($(($op:ty, $method:ident)),+ $(,)?) => {
        $(
            impl UnaryValue for $op {
                fn apply<T: RealField>(x: T) -> T {
                    x.$method()
                }
            }
        )+
    };
}

impl_direct_unary_value!(
    (ExpOp, exp),
    (LnOp, ln),
    (SinOp, sin),
    (CosOp, cos),
    (SqrtOp, sqrt),
    (TanOp, tan),
    (AsinOp, asin),
    (AcosOp, acos),
    (AtanOp, atan),
    (SinhOp, sinh),
    (CoshOp, cosh),
    (Log2Op, log2),
    (Log10Op, log10),
    (Exp2Op, exp2),
    (AtanhOp, atanh),
    (AsinhOp, asinh),
    (AcoshOp, acosh),
    (Expm1Op, exp_m1),
    (Log1pOp, ln_1p),
    (FloorOp, floor),
    (CeilOp, ceil),
    (TruncOp, trunc),
    (RoundOp, round_ties_even),
);

impl UnaryValue for AbsOp {
    fn apply<T: RealField>(x: T) -> T {
        <T as NumericElement>::abs(x)
    }
}

impl UnaryValue for NegOp {
    fn apply<T: RealField>(x: T) -> T {
        -x
    }
}

impl UnaryValue for RecipOp {
    fn apply<T: RealField>(x: T) -> T {
        x.recip()
    }
}

impl UnaryValue for IdentityOp {
    fn apply<T: RealField>(x: T) -> T {
        x
    }
}

impl UnaryValue for ExpNegOp {
    fn apply<T: RealField>(x: T) -> T {
        (-x).exp()
    }
}

/// `0` for `±0` and `NaN`, `1`/`-1` otherwise — matching the WGSL/CUDA
/// renderings (ADR 0061 Decision 7), not eunomia's `signum` (`FloatElement`),
/// which returns a signed `1.0` for `±0` and propagates `NaN`.
impl UnaryValue for SignOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            <T as NumericElement>::ZERO
        } else if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else if x < <T as NumericElement>::ZERO {
            -<T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_matches_the_documented_special_cases() {
        assert_eq!(<SignOp as UnaryValue>::apply(0.0f32), 0.0);
        assert_eq!(<SignOp as UnaryValue>::apply(-0.0f32), 0.0);
        assert_eq!(<SignOp as UnaryValue>::apply(f32::NAN), 0.0);
        assert_eq!(<SignOp as UnaryValue>::apply(3.5f32), 1.0);
        assert_eq!(<SignOp as UnaryValue>::apply(-3.5f32), -1.0);
    }

    #[test]
    fn host_blanket_reaches_unary_value_under_the_source_bound() {
        fn value_of<Op: UnaryExpr<Host>>(x: f32) -> Option<f32> {
            Op::value(x)
        }
        assert_eq!(value_of::<ExpOp>(0.0), Some(1.0));
        assert_eq!(<ExpOp as UnaryExpr<Host>>::EXPR, "host");
    }
}
