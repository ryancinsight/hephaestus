//! Host value functions for the runtime-parameter markers (ADR 0061).
//!
//! Every `apply` below reproduces its `Wgsl`/`CudaC`/`HipC` `EXPR` exactly,
//! nested-`select`/ternary order included: [`HardtanhOp`] in particular is not
//! a plain `clamp(x, first, second)`, because a reversed pair (`first >
//! second`) makes the two forms diverge (`clamp` panics or reorders; the
//! nested form does not — see this module's boundary tests).

use super::ParameterizedUnaryValue;
use super::marker::{
    CeluGradOp, CeluOp, HardshrinkGradOp, HardshrinkOp, HardtanhGradOp, HardtanhOp,
    LeakyReluGradOp, LeakyReluOp, SoftshrinkGradOp, SoftshrinkOp, ThresholdGradOp, ThresholdOp,
};
use eunomia::{NumericElement, RealField};

impl ParameterizedUnaryValue for HardtanhOp {
    fn apply<T: RealField>(x: T, first: T, second: T) -> T {
        let inner = if x > second { second } else { x };
        if x < first { first } else { inner }
    }
}

/// Open interval on both ends (ADR 0061): the boundary itself reports no
/// gradient, matching `(x > first) && (x < second)`.
impl ParameterizedUnaryValue for HardtanhGradOp {
    fn apply<T: RealField>(x: T, first: T, second: T) -> T {
        if x > first && x < second {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict `x > first` selects `x`; the boundary and below select `second`,
/// matching `select(second, x, x > first)`.
impl ParameterizedUnaryValue for ThresholdOp {
    fn apply<T: RealField>(x: T, first: T, second: T) -> T {
        if x > first { x } else { second }
    }
}

/// Strict `x > first`, matching `select(0.0, 1.0, x > first)`.
impl ParameterizedUnaryValue for ThresholdGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x > first {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Non-strict `x >= 0` selects the identity branch, matching
/// `select(first * x, x, x >= 0.0)`; both signed zeros take the identity
/// branch.
impl ParameterizedUnaryValue for LeakyReluOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            first * x
        }
    }
}

/// Strict `x > 0.0`, matching `select(first, 1.0, x > 0.0)`: both signed
/// zeros select the negative-slope branch (unlike [`LeakyReluOp`]'s
/// non-strict forward pass).
impl ParameterizedUnaryValue for LeakyReluGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            first
        }
    }
}

/// Strict `abs(x) > first`, matching `select(0.0, x, abs(x) > first)`.
impl ParameterizedUnaryValue for HardshrinkOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if <T as NumericElement>::abs(x) > first {
            x
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict `abs(x) > first`, matching `select(0.0, 1.0, abs(x) > first)`.
impl ParameterizedUnaryValue for HardshrinkGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if <T as NumericElement>::abs(x) > first {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict on both branches, matching
/// `select(select(0.0, x - first, x > first), x + first, x < -first)`.
impl ParameterizedUnaryValue for SoftshrinkOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x < -first {
            x + first
        } else if x > first {
            x - first
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict on both branches, matching
/// `select(0.0, 1.0, (x > first) || (x < -first))`.
impl ParameterizedUnaryValue for SoftshrinkGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x > first || x < -first {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// `celu(x) = x` for `x ≥ 0`, `first · expm1(x / first)` for `x < 0` (`first`
/// is α) — `expm1` rather than `exp(x / first) - 1` keeps the negative
/// branch accurate near zero (ADR 0061 Decision 6, the same cancellation
/// class as [`crate::EluOp`]).
impl ParameterizedUnaryValue for CeluOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            first * (x / first).exp_m1()
        }
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7).
impl ParameterizedUnaryValue for CeluGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            (x / first).exp()
        }
    }
}
