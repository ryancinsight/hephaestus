//! Device-neutral n-D concatenation over strided views.
//!
//! [`crate::CatOps`] writes the concatenation of strided inputs along one
//! existing axis into a caller-owned strided output: every input must share
//! the same shape on all axes except `axis`, and the output's `axis` extent
//! is the sum of the inputs' extents there. This mirrors `leto::concat`'s
//! contract so host and device results are directly comparable, while the
//! device kernel reads/writes through the same [`crate::StridedView`] every
//! other accelerator seam accepts — transposed, sliced, and offset views
//! dispatch without a contiguous staging copy.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D concatenation.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `C: CatOps<D, T>` costs nothing at runtime and every call monomorphizes to
/// the backend's own kernel dispatch.
///
/// # Shape
///
/// Every input must share the same shape on all axes except `axis`.
/// `output.layout.shape()[d]` must equal the inputs' shared shape on every
/// axis `d != axis`, and the sum of the inputs' `axis` extents on `axis`.
///
/// # Special values
///
/// Concatenation is a bitwise copy; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait CatOps<D: ComputeDevice, T: Pod> {
    /// Write the concatenation of `inputs` along `axis` into `output`.
    ///
    /// The output buffer must not alias any input buffer: every thread
    /// reads its input while other threads (and later passes) write the
    /// output, so sharing storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `inputs` is empty, `axis` is out
    /// of range, the inputs disagree on a non-`axis` dimension, `output`'s
    /// shape does not match the concatenated shape, the output buffer
    /// aliases an input buffer, a layout is unsupported, or the backend
    /// dispatch fails.
    fn cat_into<const N: usize>(
        &self,
        device: &D,
        inputs: &[StridedView<'_, D::Buffer<T>, N>],
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the concatenation geometry, returning the per-input base offset
/// along `axis` each kernel needs to place its block.
///
/// The checks run in a fixed order — `axis` range, non-empty inputs,
/// non-`axis` agreement, checked `axis` summation, output shape — so a
/// caller bug reports deterministically regardless of backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= N`,
/// `input_shapes` is empty, two inputs disagree on a non-`axis` dimension,
/// the `axis` extents overflow `usize`, or `output_shape` is not the
/// concatenated shape.
pub fn validate_cat_shape<const N: usize>(
    input_shapes: &[[usize; N]],
    axis: usize,
    output_shape: [usize; N],
) -> Result<Vec<usize>> {
    if axis >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("cat axis {axis} is out of bounds for rank {N}"),
        });
    }
    let Some(first) = input_shapes.first() else {
        return Err(HephaestusError::InvalidConfiguration {
            message: "cat requires at least one input".to_string(),
        });
    };

    let mut bases = Vec::with_capacity(input_shapes.len());
    let mut axis_total = 0usize;
    for (index, shape) in input_shapes.iter().enumerate() {
        for d in 0..N {
            if d != axis && shape[d] != first[d] {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!(
                        "cat input {index} shape mismatch at axis {d}: expected {}, got {}",
                        first[d], shape[d]
                    ),
                });
            }
        }
        bases.push(axis_total);
        axis_total = axis_total.checked_add(shape[axis]).ok_or_else(|| {
            HephaestusError::InvalidConfiguration {
                message: format!("cat axis extents overflow usize at input {index}"),
            }
        })?;
    }

    for d in 0..N {
        let expected = if d == axis { axis_total } else { first[d] };
        if output_shape[d] != expected {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "cat output shape mismatch at axis {d}: expected {expected}, got {}",
                    output_shape[d]
                ),
            });
        }
    }
    Ok(bases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_exact_concatenated_shape_with_bases() {
        let bases =
            validate_cat_shape(&[[2, 3], [2, 1], [2, 4]], 1, [2, 8]).expect("valid cat geometry");
        assert_eq!(bases, vec![0, 3, 4]);
    }

    #[test]
    fn accepts_a_single_input_as_identity_with_zero_base() {
        let bases = validate_cat_shape(&[[2, 3]], 0, [2, 3]).expect("single input is valid");
        assert_eq!(bases, vec![0]);
    }

    #[test]
    fn accepts_zero_length_segments() {
        let bases = validate_cat_shape(&[[2, 0], [2, 5]], 1, [2, 5])
            .expect("empty segments contribute nothing");
        assert_eq!(bases, vec![0, 0]);
    }

    #[test]
    fn rejects_an_axis_out_of_range() {
        let err = validate_cat_shape(&[[2, 3]], 2, [2, 3]).expect_err("axis out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_input_list() {
        let err = validate_cat_shape::<2>(&[], 0, [2, 3]).expect_err("empty input list");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_non_axis_shape_mismatch() {
        let err = validate_cat_shape(&[[2, 3], [4, 1]], 1, [2, 4]).expect_err("non-axis mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_missing_a_segment() {
        let err = validate_cat_shape(&[[2, 3], [2, 1]], 1, [2, 3]).expect_err("short output axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_axis_extents_that_overflow_usize() {
        let err =
            validate_cat_shape(&[[usize::MAX, 1], [1, 1]], 0, [0, 1]).expect_err("axis overflow");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
