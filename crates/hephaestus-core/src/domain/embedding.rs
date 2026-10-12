//! Device-neutral embedding-table gather.
//!
//! `table` is a flat `[num_embeddings, embedding_dim]` row-major buffer;
//! `indices` (length `n`) selects `n` rows, written contiguously into
//! `output` (`[n, embedding_dim]`). Out-of-range indices are a data-dependent
//! condition this seam must not silently mask (error-handling restraint): a
//! valid index is never known at prepare time (it lives in the same
//! device-resident buffer the gather reads), so validation is part of the
//! dispatch itself rather than a host-side pre-check, and implementors
//! report an out-of-range index as a typed error rather than clamping or
//! wrapping it.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};

/// Device-neutral embedding-table gather.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `E: EmbeddingOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait EmbeddingOps<D: ComputeDevice, T: Pod> {
    /// Gather `output[i] = table[indices[i]]` (each a row of `embedding_dim`
    /// elements) for every `i`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the buffer lengths do not agree
    /// with `num_embeddings`/`embedding_dim`, when any index in `indices` is
    /// `>= num_embeddings`, or when the backend dispatch fails.
    fn gather_into(
        &self,
        device: &D,
        table: &D::Buffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &D::Buffer<u32>,
        output: &D::Buffer<T>,
    ) -> Result<()>;

    /// Accumulate `grad_table[indices[i]] += grad_out[i]` (each a row of
    /// `embedding_dim` elements) for every `i`: the backward pass of
    /// [`gather_into`](Self::gather_into) over the same indices.
    ///
    /// The gradients add onto the existing contents of `grad_table`; a
    /// caller wanting pure per-table gradients zeroes it first.
    ///
    /// # Accumulation order
    ///
    /// Repeated indices make the sum order-observable in float rounding,
    /// so every implementor accumulates each output row in ascending input
    /// order (`i = 0..n`). The order is part of this contract, not a
    /// backend detail: all backends agree bitwise, not merely within test
    /// tolerance.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the buffer lengths do not agree
    /// with `num_embeddings`/`embedding_dim`, when any index in `indices` is
    /// `>= num_embeddings`, or when the backend dispatch fails.
    fn scatter_add_into(
        &self,
        device: &D,
        grad_table: &D::Buffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &D::Buffer<u32>,
        grad_out: &D::Buffer<T>,
    ) -> Result<()>;
}

/// Validate `table`/`output` lengths against `num_embeddings`,
/// `embedding_dim`, and `indices_len`, returning the row count.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when any length
/// disagrees.
pub fn validate_embedding_gather_shape(
    table_len: usize,
    num_embeddings: usize,
    embedding_dim: usize,
    indices_len: usize,
    output_len: usize,
) -> Result<usize> {
    validate_embedding_shapes(
        table_len,
        num_embeddings,
        embedding_dim,
        indices_len,
        output_len,
        "embedding table",
        "embedding output",
    )
}

/// Validate `grad_table`/`grad_out` lengths against `num_embeddings`,
/// `embedding_dim`, and `indices_len`, returning the row count.
///
/// The scatter counterpart of
/// [`validate_embedding_gather_shape`]: the arithmetic is identical, only
/// the operand roles differ.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when any length
/// disagrees.
pub fn validate_embedding_scatter_shape(
    grad_table_len: usize,
    num_embeddings: usize,
    embedding_dim: usize,
    indices_len: usize,
    grad_out_len: usize,
) -> Result<usize> {
    validate_embedding_shapes(
        grad_table_len,
        num_embeddings,
        embedding_dim,
        indices_len,
        grad_out_len,
        "embedding gradient table",
        "embedding gradient source",
    )
}

fn validate_embedding_shapes(
    table_len: usize,
    num_embeddings: usize,
    embedding_dim: usize,
    indices_len: usize,
    other_len: usize,
    table_role: &str,
    other_role: &str,
) -> Result<usize> {
    let expected_table_len = num_embeddings.checked_mul(embedding_dim).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "num_embeddings * embedding_dim overflows".to_string(),
        }
    })?;
    if table_len != expected_table_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "{table_role} length {table_len} does not match \
                 num_embeddings {num_embeddings} * embedding_dim {embedding_dim}"
            ),
        });
    }
    let expected_other_len = indices_len.checked_mul(embedding_dim).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "indices_len * embedding_dim overflows".to_string(),
        }
    })?;
    if other_len != expected_other_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "{other_role} length {other_len} does not match \
                 indices_len {indices_len} * embedding_dim {embedding_dim}"
            ),
        });
    }
    Ok(indices_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_matching_lengths() {
        assert_eq!(
            validate_embedding_gather_shape(12, 4, 3, 5, 15).expect("valid"),
            5
        );
    }

    #[test]
    fn rejects_a_table_length_mismatch() {
        let err =
            validate_embedding_gather_shape(11, 4, 3, 5, 15).expect_err("table length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_length_mismatch() {
        let err =
            validate_embedding_gather_shape(12, 4, 3, 5, 14).expect_err("output length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn scatter_accepts_matching_lengths() {
        assert_eq!(
            validate_embedding_scatter_shape(12, 4, 3, 5, 15).expect("valid"),
            5
        );
    }

    #[test]
    fn scatter_rejects_a_gradient_table_length_mismatch() {
        let err = validate_embedding_scatter_shape(11, 4, 3, 5, 15)
            .expect_err("gradient table length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn scatter_rejects_a_gradient_source_length_mismatch() {
        let err = validate_embedding_scatter_shape(12, 4, 3, 5, 14)
            .expect_err("gradient source length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
