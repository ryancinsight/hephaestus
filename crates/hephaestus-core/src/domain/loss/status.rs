use crate::{HephaestusError, Result};

/// Backend-neutral semantic preflight result written by cross-entropy kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CrossEntropyStatus {
    /// The request is valid.
    Valid = 0,
    /// A logit is NaN or infinite.
    NonFiniteLogits = 1,
    /// A target index is outside the class dimension.
    TargetOutOfRange = 2,
    /// Forward arithmetic produced NaN or infinity.
    NonFiniteForwardArithmetic = 3,
    /// The upstream scalar is NaN or infinite.
    NonFiniteOutputGradient = 4,
    /// Saved probabilities are not finite normalized rows.
    InvalidProbabilities = 5,
    /// The additive gradient destination contains NaN or infinity.
    NonFiniteGradientDestination = 6,
    /// Backward arithmetic produced NaN or infinity.
    NonFiniteBackwardArithmetic = 7,
}

impl CrossEntropyStatus {
    /// Return the stable device protocol code.
    #[must_use]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Convert a device protocol code into the canonical typed result.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] for a recognized
    /// semantic failure and [`HephaestusError::DispatchFailed`] for an unknown
    /// code, which indicates provider protocol drift.
    pub fn check(code: u32) -> Result<()> {
        let message = match code {
            code if code == Self::Valid.code() => return Ok(()),
            code if code == Self::NonFiniteLogits.code() => {
                "cross-entropy logits contain a non-finite value"
            }
            code if code == Self::TargetOutOfRange.code() => {
                "cross-entropy target is outside the class dimension"
            }
            code if code == Self::NonFiniteForwardArithmetic.code() => {
                "cross-entropy forward arithmetic produced a non-finite value"
            }
            code if code == Self::NonFiniteOutputGradient.code() => {
                "cross-entropy output gradient contains a non-finite value"
            }
            code if code == Self::InvalidProbabilities.code() => {
                "cross-entropy saved probabilities do not form a valid row"
            }
            code if code == Self::NonFiniteGradientDestination.code() => {
                "cross-entropy logit-gradient destination contains a non-finite value"
            }
            code if code == Self::NonFiniteBackwardArithmetic.code() => {
                "cross-entropy backward arithmetic produced a non-finite value"
            }
            unknown => {
                return Err(HephaestusError::DispatchFailed {
                    message: format!(
                        "cross-entropy preflight returned unknown status code {unknown}"
                    ),
                });
            }
        };
        Err(HephaestusError::InvalidConfiguration {
            message: message.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_codes_and_unknown_status_are_stable() {
        assert_eq!(CrossEntropyStatus::Valid.code(), 0);
        assert_eq!(CrossEntropyStatus::NonFiniteBackwardArithmetic.code(), 7);
        assert!(CrossEntropyStatus::check(0).is_ok());
        assert!(matches!(
            CrossEntropyStatus::check(8),
            Err(HephaestusError::DispatchFailed { .. })
        ));
    }
}
