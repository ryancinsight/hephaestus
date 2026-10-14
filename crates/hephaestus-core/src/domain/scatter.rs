//! Device-neutral n-D scatter-add over strided views.
//!
//! [`crate::ScatterOps`] writes `input` into a caller-owned strided output
//! and then accumulates `output[o] += src[i]` for every logical index `i`
//! of the source, where `o` is `i` with the `dim` lane replaced by
//! `index[i]`. Input and output share one shape; index and source share
//! one shape whose every extent fits inside the input's. This is the
//! `torch.scatter_add` semantic `coeus_ops::scatter_add` documents — the
//! backward operator of [`crate::GatherOps`] — while the device kernel
//! reads/writes through the same [`crate::StridedView`] every other
//! accelerator seam accepts.
//!
//! Accumulation order is part of the contract: each output cell sums its
//! contributing source cells in ascending row-major source order, so every backend
//! agrees bitwise on floating-point sums. An input-parallel scatter would
//! race on repeated indices (and WGSL has no floating-point atomics), so
//! device backends scan output-parallel instead, exactly as the embedding
//! scatter-add does.
//!
//! Indices are `u32`, matching [`crate::EmbeddingOps`]: the index width is a
//! seam-level choice, not a caller decision.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D scatter-add.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `S: ScatterOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input and output share rank `N` and shape. Index and source share rank
/// `N` and shape, and every non-`dim` index extent must fit inside the
/// input's extent on the same dimension; the `dim` extent counts updates
/// and may exceed it.
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
/// Scattering adds `T` values; the only scalar operation is the seam's own
/// accumulation in ascending source order, so no special-value handling
/// applies beyond what the scalar type itself represents.
pub trait ScatterOps<D: ComputeDevice, T: Pod> {
    /// Write `input` into `output`, then accumulate `src[i]` into
    /// `output[i with dim := index[i]]` in ascending row-major source order.
    ///
    /// The output buffer must not alias the input, index, or source
    /// buffer: accumulation reads all three while writing the output, so
    /// sharing storage would race. (The index buffer holds `u32` against
    /// the output's `T`, so those two cannot name one buffer; the
    /// input-output and source-output pairs are checked.)
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `dim` is out of range, index
    /// and source shapes disagree, an index extent exceeds the input's,
    /// the output shape does not match the input shape, any index value
    /// is out of range for the input's `dim` extent, the output buffer
    /// aliases the input or source buffer, a layout is unsupported, or the
    /// backend dispatch fails.
    fn scatter_add_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        dim: usize,
        index: StridedView<'_, D::Buffer<u32>, N>,
        src: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the scatter-add geometry: `dim` range, index/source shape
/// agreement, non-`dim` index extents inside the input's, and
/// output-equals-input shape.
///
/// The checks run in a fixed order — `dim` range, index/source agreement,
/// index-inside-input extents, output shape — so a caller bug reports
/// deterministically regardless of backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `dim >= N`, two
/// index/source extents disagree, a non-`dim` index extent exceeds the
/// input's, or any output dimension misses the input shape.
pub fn validate_scatter_shape<const N: usize>(
    input_shape: [usize; N],
    dim: usize,
    index_shape: [usize; N],
    src_shape: [usize; N],
    output_shape: [usize; N],
) -> Result<()> {
    if dim >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("scatter dim {dim} is out of bounds for rank {N}"),
        });
    }
    for (d, (&index_extent, &src_extent)) in index_shape.iter().zip(src_shape.iter()).enumerate() {
        if index_extent != src_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "scatter index/src shape mismatch at axis {d}: expected {index_extent}, got {src_extent}",
                ),
            });
        }
    }
    // Only non-`dim` extents are positions that must fit: the `dim` extent
    // counts updates, and repeated indices accumulate in ascending source
    // order, so it may exceed the input's `dim` extent freely.
    for (d, (&index_extent, &input_extent)) in
        index_shape.iter().zip(input_shape.iter()).enumerate()
    {
        if d != dim && index_extent > input_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "scatter index extent {index_extent} at axis {d} exceeds the input extent {input_extent}",
                ),
            });
        }
    }
    for (d, (&output_extent, &input_extent)) in
        output_shape.iter().zip(input_shape.iter()).enumerate()
    {
        if output_extent != input_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "scatter output shape mismatch at axis {d}: expected {input_extent}, got {output_extent}",
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
    fn accepts_the_exact_scatter_geometry() {
        validate_scatter_shape([2, 4], 1, [2, 3], [2, 3], [2, 4]).expect("valid scatter geometry");
    }

    #[test]
    fn accepts_a_narrower_non_dim_index_extent() {
        validate_scatter_shape([3, 4], 1, [2, 3], [2, 3], [3, 4]).expect("narrow index is valid");
    }

    #[test]
    fn accepts_a_dim_extent_wider_than_the_input() {
        // The dim extent counts updates, not positions: seven updates into
        // two slots is legal (repeated indices accumulate in order).
        validate_scatter_shape([3, 2], 1, [3, 7], [3, 7], [3, 2])
            .expect("wide dim extent is valid");
    }

    #[test]
    fn accepts_an_empty_index() {
        validate_scatter_shape([2, 4], 1, [2, 0], [2, 0], [2, 4])
            .expect("vacuous scatter is valid");
    }

    #[test]
    fn rejects_a_dim_out_of_range() {
        let err = validate_scatter_shape([2, 4], 2, [2, 3], [2, 3], [2, 4])
            .expect_err("dim out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_index_src_shape_mismatch() {
        let err = validate_scatter_shape([2, 4], 1, [2, 3], [2, 2], [2, 4])
            .expect_err("index/src mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_index_extent_outside_the_input() {
        let err = validate_scatter_shape([2, 4], 1, [5, 3], [5, 3], [2, 4])
            .expect_err("index outside input");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_missing_the_input_shape() {
        let err =
            validate_scatter_shape([2, 4], 1, [2, 3], [2, 3], [2, 3]).expect_err("output mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
