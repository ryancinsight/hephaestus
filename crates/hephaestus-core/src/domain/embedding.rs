//! Device-neutral embedding-table gather and its scatter-add backward.
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
//!
//! The backward pass ([`crate::EmbeddingOps::scatter_add_into`]) accumulates each
//! upstream row `grad_output[i]` into `grad_table[indices[i]]`, in place.
//! Accumulation order is part of the contract: every output cell sums its
//! contributing rows in ascending index-position order, so all backends
//! agree bitwise on floating-point sums. An input-parallel scatter would
//! race on repeated indices (and WGSL has no floating-point atomics), so
//! device backends scan output-parallel instead — one thread per table row
//! sweeping all `n` positions — exactly as [`crate::ScatterOps`] does.

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

    /// Scatter-add the upstream gradient into the table gradient, in place:
    /// `grad_table[indices[i]] += grad_output[i]` (each a row of
    /// `embedding_dim` elements) for every `i`, accumulating into whatever
    /// `grad_table` held on entry (zero for a fresh backward pass).
    ///
    /// Each table cell sums its contributing rows in ascending index-position
    /// order, so repeated indices accumulate deterministically and every
    /// backend agrees bitwise.
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
        grad_output: &D::Buffer<T>,
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
    let expected_table_len = num_embeddings.checked_mul(embedding_dim).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "num_embeddings * embedding_dim overflows".to_string(),
        }
    })?;
    if table_len != expected_table_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "embedding table length {table_len} does not match \
                 num_embeddings {num_embeddings} * embedding_dim {embedding_dim}"
            ),
        });
    }
    let expected_output_len = indices_len.checked_mul(embedding_dim).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "indices_len * embedding_dim overflows".to_string(),
        }
    })?;
    if output_len != expected_output_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "embedding output length {output_len} does not match \
                 indices_len {indices_len} * embedding_dim {embedding_dim}"
            ),
        });
    }
    Ok(indices_len)
}

/// Validate `grad_table`/`grad_output` lengths against `num_embeddings`,
/// `embedding_dim`, and `indices_len`, returning the update count.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when any length
/// disagrees.
pub fn validate_embedding_scatter_add_shape(
    grad_table_len: usize,
    num_embeddings: usize,
    embedding_dim: usize,
    indices_len: usize,
    grad_output_len: usize,
) -> Result<usize> {
    let expected_table_len = num_embeddings.checked_mul(embedding_dim).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "num_embeddings * embedding_dim overflows".to_string(),
        }
    })?;
    if grad_table_len != expected_table_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "embedding grad-table length {grad_table_len} does not match \
                 num_embeddings {num_embeddings} * embedding_dim {embedding_dim}"
            ),
        });
    }
    let expected_grad_len = indices_len.checked_mul(embedding_dim).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "indices_len * embedding_dim overflows".to_string(),
        }
    })?;
    if grad_output_len != expected_grad_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "embedding grad-output length {grad_output_len} does not match \
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
    fn scatter_add_accepts_matching_lengths() {
        assert_eq!(
            validate_embedding_scatter_add_shape(12, 4, 3, 5, 15).expect("valid"),
            5
        );
    }

    #[test]
    fn scatter_add_rejects_a_table_length_mismatch() {
        let err = validate_embedding_scatter_add_shape(11, 4, 3, 5, 15)
            .expect_err("table length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn scatter_add_rejects_a_grad_length_mismatch() {
        let err = validate_embedding_scatter_add_shape(12, 4, 3, 5, 14)
            .expect_err("grad length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn scatter_add_rejects_a_dimension_overflow() {
        let err = validate_embedding_scatter_add_shape(0, usize::MAX, 2, 0, 0)
            .expect_err("dimension overflow");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
