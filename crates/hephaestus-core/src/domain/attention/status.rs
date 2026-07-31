use crate::{HephaestusError, Result};

/// Device-reported semantic validation outcome for attention dispatch.
///
/// Providers write one of these stable integer codes during a read-only
/// preflight. The host checks the code before any caller-owned destination is
/// mutated, preserving failure atomicity without downloading tensor operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum AttentionSemanticStatus {
    /// Every semantic precondition holds.
    Valid = 0,
    /// Query contains a non-finite value.
    NonFiniteQuery = 1,
    /// Key contains a non-finite value.
    NonFiniteKey = 2,
    /// Value contains a non-finite value.
    NonFiniteValue = 3,
    /// Keep mask contains a non-finite value.
    NonFiniteKeep = 4,
    /// Output gradient contains a non-finite value.
    NonFiniteOutputGradient = 5,
    /// Saved attention weights contain a non-finite value.
    NonFiniteWeights = 6,
    /// Saved attention weights do not form a probability row.
    InvalidWeights = 7,
    /// A selected query-gradient destination contains a non-finite value.
    NonFiniteQueryGradient = 8,
    /// A selected key-gradient destination contains a non-finite value.
    NonFiniteKeyGradient = 9,
    /// A selected value-gradient destination contains a non-finite value.
    NonFiniteValueGradient = 10,
    /// Forward score or probability arithmetic produced a non-finite value.
    NonFiniteWeightsArithmetic = 11,
    /// Forward output arithmetic produced a non-finite value.
    NonFiniteOutputArithmetic = 12,
    /// Query-gradient arithmetic or accumulation produced a non-finite value.
    NonFiniteQueryGradientArithmetic = 13,
    /// Key-gradient arithmetic or accumulation produced a non-finite value.
    NonFiniteKeyGradientArithmetic = 14,
    /// Value-gradient arithmetic or accumulation produced a non-finite value.
    NonFiniteValueGradientArithmetic = 15,
}

impl AttentionSemanticStatus {
    /// Return the stable device ABI code.
    #[must_use]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decode and validate a device-written status code.
    ///
    /// # Errors
    ///
    /// Returns a dispatch error for an unknown code or the typed semantic
    /// failure represented by a known nonzero code.
    pub fn check(code: u32) -> Result<()> {
        let message = match code {
            code if code == Self::Valid.code() => return Ok(()),
            code if code == Self::NonFiniteQuery.code() => {
                "attention query contains a non-finite value"
            }
            code if code == Self::NonFiniteKey.code() => {
                "attention key contains a non-finite value"
            }
            code if code == Self::NonFiniteValue.code() => {
                "attention value contains a non-finite value"
            }
            code if code == Self::NonFiniteKeep.code() => {
                "attention keep mask contains a non-finite value"
            }
            code if code == Self::NonFiniteOutputGradient.code() => {
                "attention output gradient contains a non-finite value"
            }
            code if code == Self::NonFiniteWeights.code() => {
                "attention weights contain a non-finite value"
            }
            code if code == Self::InvalidWeights.code() => {
                "attention weights do not form a probability row"
            }
            code if code == Self::NonFiniteQueryGradient.code() => {
                "attention query-gradient destination contains a non-finite value"
            }
            code if code == Self::NonFiniteKeyGradient.code() => {
                "attention key-gradient destination contains a non-finite value"
            }
            code if code == Self::NonFiniteValueGradient.code() => {
                "attention value-gradient destination contains a non-finite value"
            }
            code if code == Self::NonFiniteWeightsArithmetic.code() => {
                "attention weight arithmetic produced a non-finite value"
            }
            code if code == Self::NonFiniteOutputArithmetic.code() => {
                "attention output arithmetic produced a non-finite value"
            }
            code if code == Self::NonFiniteQueryGradientArithmetic.code() => {
                "attention query-gradient arithmetic produced a non-finite value"
            }
            code if code == Self::NonFiniteKeyGradientArithmetic.code() => {
                "attention key-gradient arithmetic produced a non-finite value"
            }
            code if code == Self::NonFiniteValueGradientArithmetic.code() => {
                "attention value-gradient arithmetic produced a non-finite value"
            }
            unknown => {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("attention preflight returned unknown status code {unknown}"),
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
    fn status_codes_are_stable_and_unknown_codes_are_device_failures() {
        assert_eq!(AttentionSemanticStatus::Valid.code(), 0);
        assert_eq!(
            AttentionSemanticStatus::NonFiniteValueGradientArithmetic.code(),
            15
        );
        assert!(AttentionSemanticStatus::check(0).is_ok());
        assert_eq!(
            AttentionSemanticStatus::check(16)
                .expect_err("unknown status must fail")
                .to_string(),
            "kernel dispatch failed: attention preflight returned unknown status code 16"
        );
    }
}
