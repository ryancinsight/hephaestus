//! Device-neutral batched 3-vector cross product.
//!
//! Companion to [`crate::DenseVectorOps`]'s `dot`/`norm_l2` reductions:
//! those already dispatch on-device (only their scalar result crosses back
//! to host), but no seam covered the other common dense-vector primitive —
//! cross product, whose output is itself a vector, not a scalar. Buffers are
//! flat and contiguous, matching `DenseVectorOps`'s own convention rather
//! than `StridedView`'s generality: a cross product is defined triple-wise
//! over consecutive `(x, y, z)` elements, so a caller with a non-contiguous
//! batch of 3-vectors materializes it contiguously first, exactly as
//! `DenseVectorOps` already expects for `dot`/`norm_l2`.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};

/// Device-neutral batched cross product over flat arrays of 3-vectors.
///
/// `a`, `b`, and `out` each hold `n` consecutive `(x, y, z)` triples
/// (length `3 * n`); `out[3*i..3*i+3] = cross(a[3*i..3*i+3], b[3*i..3*i+3])`
/// for every `i`.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `C: CrossProductOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait CrossProductOps<D: ComputeDevice, T: Pod> {
    /// Write the batched cross product of `a` and `b` into `out`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `a`, `b`, and `out` do not share
    /// one length that is a multiple of three, when `out` aliases `a` or
    /// `b`, or when the backend dispatch fails.
    fn cross_into(
        &self,
        device: &D,
        a: &D::Buffer<T>,
        b: &D::Buffer<T>,
        out: &D::Buffer<T>,
    ) -> Result<()>;
}

/// Validate that `a_len`, `b_len`, and `out_len` agree and are each a
/// multiple of three, returning the triple count.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when the lengths
/// disagree or are not a multiple of three.
pub fn validate_cross_product_lengths(a_len: usize, b_len: usize, out_len: usize) -> Result<usize> {
    if a_len != b_len || a_len != out_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "cross product operand lengths must match: a={a_len}, b={b_len}, out={out_len}"
            ),
        });
    }
    if !a_len.is_multiple_of(3) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("cross product operand length {a_len} is not a multiple of three"),
        });
    }
    Ok(a_len / 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_matching_multiple_of_three_lengths() {
        assert_eq!(validate_cross_product_lengths(6, 6, 6).expect("valid"), 2);
    }

    #[test]
    fn rejects_a_length_mismatch() {
        let err = validate_cross_product_lengths(6, 3, 6).expect_err("mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_non_multiple_of_three() {
        let err = validate_cross_product_lengths(4, 4, 4).expect_err("not a multiple of 3");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
