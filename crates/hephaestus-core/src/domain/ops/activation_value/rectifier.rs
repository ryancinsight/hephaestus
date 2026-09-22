//! Value functions for the rectifier family: ReLU and ELU, forward and gradient.

use super::super::{EluGradOp, EluOp, ReluGradOp, ReluOp, UnaryValue};
use eunomia::{NumericElement, RealField};

/// `max(x, 0)` via an explicit `is_nan` check rather than `T::max` alone:
/// eunomia's `max_scalar` contract has a single `NaN` operand *ignored*, so
/// `NaN.max(0) == 0` — silently discarding the `NaN` instead of propagating
/// it, the opposite of IEEE 754's own `maxNum`/comparison-predicate
/// semantics for a unary activation.
impl UnaryValue for ReluOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            x
        } else {
            x.max(<T as NumericElement>::ZERO)
        }
    }
}

/// Takes the input (ADR 0061 Decision 7), matching the WGSL/CUDA rendering.
/// `x > 0` is `false` for `NaN` (IEEE 754 unordered comparisons), so the
/// naive two-arm form silently returns `0` for a `NaN` input instead of
/// propagating it; the explicit `is_nan` check comes first.
impl UnaryValue for ReluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x.is_nan() {
            x
        } else if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// `elu(x) = x` for `x ≥ 0`, `expm1(x)` for `x < 0` — `expm1` rather than
/// `exp(x) - 1` keeps the negative branch accurate near zero.
impl UnaryValue for EluOp {
    fn apply<T: RealField>(x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            x.exp_m1()
        }
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7).
impl UnaryValue for EluGradOp {
    fn apply<T: RealField>(x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            x.exp()
        }
    }
}
