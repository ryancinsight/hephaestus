//! Calibration of the fixed native probes, independent of adapter identity.
//!
//! WGSL 15.7.4.1 gives exp/exp2 `(3 + 2|x|)` ULP, log/log2 an absolute
//! bound near one and 3 ULP elsewhere, and pow the exp2(y*log2(x)) envelope.
//! See <https://www.w3.org/TR/WGSL/#floating-point-accuracy>.
//!
//! For the unchanged normal-input probes, with epsilon <= 2^-23:
//! * Exp: |x| <= 20 gives 43 epsilon.
//! * Log: |log(x)| >= 1/2 at nonzero outputs; input quantization and the
//!   near-one absolute bound give less than 9 epsilon. Zero identities retain
//!   the probe's explicit relative-error rejection.
//! * Pow: |y| <= 10, |y log2(x)| <= 10. Input quantization contributes
//!   less than 21 epsilon; log2 and multiplication less than 51 epsilon.
//!   Hence |delta t| < 72 epsilon, 2^|delta t|-1 < 73 epsilon, and exp2
//!   contributes less than 25 epsilon: total less than 99 epsilon.
//! * Tanh: the sinh/cosh quotient contributes less than 100 epsilon.
//!   For nonzero probes |tanh(x)| > 1/3, the alternative absolute allowance
//!   gives 3/100000 <= 252 epsilon at binary32. Its width-scaled counterpart
//!   defines the diagnostic model for the experimental binary64 extension.
//!
//! If GPU/reference relative envelopes are g/r, comparison against the
//! reference is bounded by (g+r)/(1-r). Binary64 is a calibration model,
//! not a WGSL binary64 conformance claim; illegal lowerings remain blocked.
//! Envelope arithmetic uses the associated reference precision so the
//! reference contribution cannot disappear in the target precision. The
//! coefficients, scaled epsilons, sum and denominator are exact binary64
//! values; rounding the final quotient upward encloses the exact ratio.

use eunomia::{FloatElement, NumericElement, RealField};

use super::verdict::MathBuiltin;

pub(super) trait ErrorScalar: RealField {
    fn upward(self) -> Self;
}

impl ErrorScalar for f64 {
    #[inline]
    fn upward(self) -> Self {
        self.next_up()
    }
}

pub(super) trait PrecisionModel: FloatElement {
    type Reference: ErrorScalar;
    fn reference_epsilon() -> Self::Reference;
    fn to_reference(self) -> Self::Reference;
}

impl PrecisionModel for f32 {
    type Reference = f64;

    #[inline]
    fn reference_epsilon() -> Self::Reference {
        // Epsilon is an exact dyadic; widening is required by error analysis.
        f64::from(f32::EPSILON)
    }

    #[inline]
    fn to_reference(self) -> Self::Reference {
        f64::from(self)
    }
}

impl PrecisionModel for f64 {
    type Reference = f64;

    #[inline]
    fn reference_epsilon() -> Self::Reference {
        Self::EPSILON
    }

    #[inline]
    fn to_reference(self) -> Self::Reference {
        self
    }
}

pub(super) const fn ulp_budget(builtin: MathBuiltin) -> usize {
    match builtin {
        MathBuiltin::Exp => 3 + 2 * 20,
        MathBuiltin::Log => 9,
        MathBuiltin::Pow => 99,
        MathBuiltin::Tanh => (3_usize * (1_usize << 23)).div_ceil(100_000),
    }
}

pub(super) fn accuracy_bound<T: PrecisionModel>(builtin: MathBuiltin) -> T::Reference {
    let budget = T::Reference::from_count(ulp_budget(builtin));
    let target = budget * T::reference_epsilon();
    let reference = budget * T::Reference::EPSILON;
    ((target + reference) / (T::Reference::ONE - reference)).upward()
}

/// Relative comparison is defined at zero only for an exact zero result.
/// Any nonzero result there, or any nonfinite operand, has unbounded error.
pub(super) fn relative_error<R: ErrorScalar>(actual: R, reference: R) -> R {
    if !actual.is_finite() || !reference.is_finite() {
        R::INFINITY
    } else if reference == R::ZERO {
        if actual == R::ZERO {
            R::ZERO
        } else {
            R::INFINITY
        }
    } else {
        (actual - reference).abs() / reference.abs()
    }
}

#[cfg(test)]
mod tests;
