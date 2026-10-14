//! Device-neutral n-D splitting over strided views.
//!
//! [`crate::SplitOps`] writes consecutive segments of one strided input
//! along one existing axis into caller-owned strided outputs: every output
//! must share the input's shape on all axes except `axis`, and the `sizes`
//! must sum to the input's `axis` extent. This mirrors `leto::split`'s
//! contract so host and device results are directly comparable, while the
//! device kernel reads/writes through the same [`crate::StridedView`] every
//! other accelerator seam accepts — transposed, sliced, and offset views
//! dispatch without a contiguous staging copy.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D splitting.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `S: SplitOps<D, T>` costs nothing at runtime and every call monomorphizes
/// to the backend's own kernel dispatch.
///
/// # Shape
///
/// Every output must share the input's shape on every axis `d != axis`, and
/// `output[i].layout.shape()[axis]` must equal `sizes[i]`. The `sizes` must
/// sum to the input's `axis` extent.
///
/// # Special values
///
/// Splitting is a bitwise copy; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait SplitOps<D: ComputeDevice, T: Pod> {
    /// Write consecutive `sizes` segments of `input` along `axis` into `outputs`.
    ///
    /// No output buffer may alias the input buffer or another output buffer:
    /// every thread reads the input while other threads (and later passes)
    /// write the outputs, so sharing storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis` is out of range,
    /// `sizes.len() != outputs.len()`, an output disagrees with the input on
    /// a non-`axis` dimension or on its `sizes` extent, the `sizes` do not
    /// sum to the input's `axis` extent, an output buffer aliases the input
    /// or another output, a layout is unsupported, or the backend dispatch
    /// fails.
    fn split_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        axis: usize,
        sizes: &[usize],
        outputs: &[StridedView<'_, D::Buffer<T>, N>],
    ) -> Result<()>;
}

/// Validate the split geometry, returning the per-segment base offset along
/// `axis` each kernel needs to place its block.
///
/// The checks run in a fixed order — `axis` range, `sizes`/outputs arity,
/// per-output shape agreement, checked `axis` summation — so a caller bug
/// reports deterministically regardless of backend.
///
/// An empty `sizes` list is legal exactly when the input's `axis` extent is
/// zero: every check passes vacuously and the bases are empty, mirroring
/// `leto::split`.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= N`,
/// `sizes.len() != output_shapes.len()`, an output disagrees with the input
/// on a non-`axis` dimension or on its `sizes` extent, the `sizes` overflow
/// `usize` while summing, or the sum is not the input's `axis` extent.
pub fn validate_split_shape<const N: usize>(
    input_shape: [usize; N],
    axis: usize,
    sizes: &[usize],
    output_shapes: &[[usize; N]],
) -> Result<Vec<usize>> {
    if axis >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("split axis {axis} is out of bounds for rank {N}"),
        });
    }
    if sizes.len() != output_shapes.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "split arity mismatch: {} sizes for {} outputs",
                sizes.len(),
                output_shapes.len()
            ),
        });
    }

    for (index, shape) in output_shapes.iter().enumerate() {
        for d in 0..N {
            let expected = if d == axis {
                sizes[index]
            } else {
                input_shape[d]
            };
            if shape[d] != expected {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!(
                        "split output {index} shape mismatch at axis {d}: expected {expected}, got {}",
                        shape[d]
                    ),
                });
            }
        }
    }

    let mut bases = Vec::with_capacity(sizes.len());
    let mut axis_total = 0usize;
    for (index, &size) in sizes.iter().enumerate() {
        bases.push(axis_total);
        axis_total =
            axis_total
                .checked_add(size)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: format!("split sizes overflow usize at segment {index}"),
                })?;
    }
    if axis_total != input_shape[axis] {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "split sizes sum to {axis_total} but axis {axis} has length {}",
                input_shape[axis]
            ),
        });
    }
    Ok(bases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_exact_split_geometry_with_bases() {
        let bases = validate_split_shape([2, 8], 1, &[3, 1, 4], &[[2, 3], [2, 1], [2, 4]])
            .expect("valid split geometry");
        assert_eq!(bases, vec![0, 3, 4]);
    }

    #[test]
    fn accepts_a_single_full_segment_as_identity_with_zero_base() {
        let bases =
            validate_split_shape([2, 3], 0, &[2], &[[2, 3]]).expect("single segment is valid");
        assert_eq!(bases, vec![0]);
    }

    #[test]
    fn accepts_zero_length_segments() {
        let bases = validate_split_shape([2, 5], 1, &[0, 5], &[[2, 0], [2, 5]])
            .expect("empty segments contribute nothing");
        assert_eq!(bases, vec![0, 0]);
    }

    #[test]
    fn accepts_empty_sizes_on_a_zero_length_axis() {
        let bases = validate_split_shape([2, 0], 1, &[], &[]).expect("vacuous split is valid");
        assert!(bases.is_empty());
    }

    #[test]
    fn rejects_an_axis_out_of_range() {
        let err = validate_split_shape([2, 3], 2, &[3], &[[2, 3]]).expect_err("axis out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_sizes_outputs_arity_mismatch() {
        let err = validate_split_shape([2, 3], 1, &[1, 2], &[[2, 3]]).expect_err("arity mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_non_axis_shape_mismatch() {
        let err = validate_split_shape([2, 8], 1, &[3, 5], &[[2, 3], [4, 5]])
            .expect_err("non-axis mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_axis_extent_missing_its_size() {
        let err = validate_split_shape([2, 8], 1, &[3, 5], &[[2, 3], [2, 4]])
            .expect_err("short output axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_sizes_that_do_not_sum_to_the_axis_length() {
        let err = validate_split_shape([2, 8], 1, &[3, 4], &[[2, 3], [2, 4]])
            .expect_err("short sizes sum");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_sizes_that_overflow_usize() {
        let err = validate_split_shape(
            [usize::MAX, 1],
            0,
            &[usize::MAX, 1],
            &[[usize::MAX, 1], [1, 1]],
        )
        .expect_err("sizes overflow");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
