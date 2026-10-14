//! Device-neutral n-D index selection over strided views.
//!
//! [`crate::IndexSelectOps`] writes `output[i] = input[j]`, where `j`
//! is `i` with the `dim` lane replaced by `index[i_dim]`, into a
//! caller-owned strided output, where `i_dim` is `i`'s own `dim`
//! coordinate: the 1-D index holds one
//! source position per output slice along `dim`. Input and output share
//! rank `N`; every non-`dim` dimension agrees, and the output's `dim`
//! extent is the index length. This is the `torch.index_select` semantic
//! `coeus_ops::index_select` documents, while the device kernel
//! reads/writes through the same [`crate::StridedView`] every other
//! accelerator seam accepts — transposed, sliced, and offset views
//! dispatch without a contiguous staging copy.
//!
//! Indices are `u32`, matching [`crate::EmbeddingOps`]: the index width is a
//! seam-level choice, not a caller decision, so every backend packs and
//! bounds-checks one width.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D index selection.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `I: IndexSelectOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input and output share rank `N`. Every non-`dim` dimension must agree;
/// the output's `dim` extent must equal the index length. The index rank
/// is concrete (`1`), so no rank-relation bound is needed.
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
/// Selection is a bitwise copy; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait IndexSelectOps<D: ComputeDevice, T: Pod> {
    /// Write `output[i] = input[i with dim := index[i_dim]]` for every `i`.
    ///
    /// The output buffer must not alias the input buffer: every thread
    /// reads the input while other threads write the output, so sharing
    /// storage would race. (The index buffer holds `u32` against the
    /// output's `T`, so the two cannot name one buffer.)
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `dim` is out of range, a
    /// non-`dim` dimension disagrees, the output's `dim` extent is not the
    /// index length, any index value is out of range for the input's `dim`
    /// extent, the output buffer aliases the input buffer, a layout is
    /// unsupported, or the backend dispatch fails.
    fn index_select_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        dim: usize,
        index: StridedView<'_, D::Buffer<u32>, 1>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the index-selection geometry: `dim` range, non-`dim`
/// agreement, and output `dim` extent against the index length.
///
/// The checks run in a fixed order — `dim` range, non-`dim` agreement,
/// `dim` extent — so a caller bug reports deterministically regardless of
/// backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `dim >= N`, two
/// shapes disagree on a non-`dim` dimension, or the output's `dim` extent
/// is not the index length.
pub fn validate_index_select_shape<const N: usize>(
    input_shape: [usize; N],
    dim: usize,
    index_len: usize,
    output_shape: [usize; N],
) -> Result<()> {
    if dim >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("index_select dim {dim} is out of bounds for rank {N}"),
        });
    }
    for (d, (&input_extent, &output_extent)) in
        input_shape.iter().zip(output_shape.iter()).enumerate()
    {
        if d != dim && input_extent != output_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "index_select input/output shape mismatch at axis {d}: expected {input_extent}, got {output_extent}",
                ),
            });
        }
    }
    if output_shape[dim] != index_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "index_select output dim extent {} is not the index length {index_len}",
                output_shape[dim]
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_exact_index_select_geometry() {
        validate_index_select_shape([2, 4], 1, 3, [2, 3]).expect("valid geometry");
    }

    #[test]
    fn accepts_a_full_range_selection() {
        validate_index_select_shape([2, 4], 1, 4, [2, 4]).expect("full range is valid");
    }

    #[test]
    fn accepts_a_repeated_wider_than_input_selection() {
        // The index length counts selections, not positions: repeats are
        // legal, so it may exceed the input's `dim` extent freely.
        validate_index_select_shape([2, 4], 1, 9, [2, 9]).expect("wide selection is valid");
    }

    #[test]
    fn accepts_an_empty_index() {
        validate_index_select_shape([2, 4], 1, 0, [2, 0]).expect("vacuous selection is valid");
    }

    #[test]
    fn rejects_a_dim_out_of_range() {
        let err = validate_index_select_shape([2, 4], 2, 3, [2, 3]).expect_err("dim out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_non_dim_shape_mismatch() {
        let err = validate_index_select_shape([2, 4], 1, 3, [5, 3]).expect_err("non-dim mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_dim_extent_missing_the_index_length() {
        let err =
            validate_index_select_shape([2, 4], 1, 3, [2, 2]).expect_err("dim extent mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
