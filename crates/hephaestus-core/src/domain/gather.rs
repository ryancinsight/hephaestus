//! Device-neutral n-D gather over strided views.
//!
//! [`crate::GatherOps`] writes `output[i] = input[i with the `dim` lane
//! replaced by `index[i]]` into a caller-owned strided output: input, index,
//! and output share one rank, every non-`dim` dimension agrees across all
//! three, and the output shape equals the index shape exactly. This is the
//! `torch.gather` semantic `coeus_ops::gather` documents, while the device
//! kernel reads/writes through the same [`crate::StridedView`] every other
//! accelerator seam accepts — transposed, sliced, and offset views dispatch
//! without a contiguous staging copy.
//!
//! Indices are `u32`, matching [`crate::EmbeddingOps`]: the index width is a
//! seam-level choice, not a caller decision, so every backend packs and
//! bounds-checks one width.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D gather.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `G: GatherOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input, index, and output share rank `N`. Every non-`dim` dimension must
/// agree across all three shapes; the output shape must equal the index
/// shape on every dimension, including `dim`.
///
/// # Out-of-range indices
///
/// An index value `>= input.shape()[dim]` is a data-dependent fault, not a
/// shape error: the host rejects it while reading, and device backends
/// report it through the same one-`u32` atomic error flag the embedding
/// seam uses, downloaded after dispatch. No backend clamps or wraps.
///
/// # Special values
///
/// Gathering is a bitwise copy; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait GatherOps<D: ComputeDevice, T: Pod> {
    /// Write `output[i] = input[i with dim := index[i]]` for every `i`.
    ///
    /// The output buffer must not alias the input buffer: every thread
    /// reads the input while other threads write the output, so sharing
    /// storage would race. (The index buffer holds `u32` against the
    /// output's `T`, so the two cannot name one buffer.)
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `dim` is out of range, a
    /// non-`dim` dimension disagrees, the output shape does not match the
    /// index shape, any index value is out of range for the input's `dim`
    /// extent, the output buffer aliases the input buffer, a layout is
    /// unsupported, or the backend dispatch fails.
    fn gather_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        dim: usize,
        index: StridedView<'_, D::Buffer<u32>, N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the gather geometry: `dim` range, non-`dim` agreement across
/// input/index/output, and output-equals-index shape.
///
/// The checks run in a fixed order — `dim` range, input/index non-`dim`
/// agreement, output shape — so a caller bug reports deterministically
/// regardless of backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `dim >= N`, two
/// shapes disagree on a non-`dim` dimension, or any output dimension
/// misses the index shape.
pub fn validate_gather_shape<const N: usize>(
    input_shape: [usize; N],
    dim: usize,
    index_shape: [usize; N],
    output_shape: [usize; N],
) -> Result<()> {
    if dim >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("gather dim {dim} is out of bounds for rank {N}"),
        });
    }
    for (d, (&input_extent, &index_extent)) in
        input_shape.iter().zip(index_shape.iter()).enumerate()
    {
        if d != dim && input_extent != index_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "gather input/index shape mismatch at axis {d}: expected {input_extent}, got {index_extent}",
                ),
            });
        }
    }
    for (d, (&output_extent, &index_extent)) in
        output_shape.iter().zip(index_shape.iter()).enumerate()
    {
        if output_extent != index_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "gather output shape mismatch at axis {d}: expected {index_extent}, got {output_extent}",
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_exact_gather_geometry() {
        validate_gather_shape([2, 4], 1, [2, 3], [2, 3]).expect("valid gather geometry");
    }

    #[test]
    fn accepts_a_gather_dim_narrower_than_the_input() {
        validate_gather_shape([3, 5], 1, [3, 1], [3, 1]).expect("narrow gather is valid");
    }

    #[test]
    fn accepts_a_gather_dim_wider_than_the_input() {
        validate_gather_shape([3, 2], 1, [3, 7], [3, 7]).expect("wide gather is valid");
    }

    #[test]
    fn accepts_an_empty_index() {
        validate_gather_shape([2, 4], 1, [2, 0], [2, 0]).expect("vacuous gather is valid");
    }

    #[test]
    fn rejects_a_dim_out_of_range() {
        let err = validate_gather_shape([2, 4], 2, [2, 3], [2, 3]).expect_err("dim out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_input_index_non_dim_mismatch() {
        let err = validate_gather_shape([2, 4], 1, [3, 3], [3, 3]).expect_err("non-dim mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_missing_the_index_shape() {
        let err = validate_gather_shape([2, 4], 1, [2, 3], [2, 4]).expect_err("output mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_non_dim_mismatch() {
        let err =
            validate_gather_shape([2, 4], 1, [2, 3], [5, 3]).expect_err("output non-dim mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
