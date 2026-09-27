//! Device-neutral rank-2 argmax/argmin over strided operands.
//!
//! Companion to [`crate::AxisReductionOps`]: that seam reduces to a *value*,
//! this one reduces to the *position* of the extremum along `axis`, written
//! as `u32` into a caller-owned output view of the same rank (the reduced
//! axis kept at length one, matching `AxisReductionOps`'s convention so the
//! result stays directly broadcastable against the input).
//!
//! Ties break toward the lowest index along the axis — the first strict
//! improvement wins — matching `leto`'s `argmax`/`argmin` tie-break rule, so
//! host and device results agree bit-for-bit on integer and exact-float
//! fixtures.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::view::StridedView;

/// Device-neutral rank-2 argmax/argmin.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `A: ArgReduceOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait ArgReduceOps<D: ComputeDevice, T: Pod + PartialOrd> {
    /// Write the index (along `axis`) of the maximum element of each lane of
    /// `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis >= 2`, `output`'s shape does
    /// not match `input`'s shape with `axis` collapsed to length one, the
    /// reduced axis is empty, or the backend dispatch fails.
    fn argmax_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<u32>, 2>,
    ) -> Result<()>;

    /// Write the index (along `axis`) of the minimum element of each lane of
    /// `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis >= 2`, `output`'s shape does
    /// not match `input`'s shape with `axis` collapsed to length one, the
    /// reduced axis is empty, or the backend dispatch fails.
    fn argmin_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<u32>, 2>,
    ) -> Result<()>;
}

/// Validate `axis`, `input_shape`, and `output_shape` for an arg-reduce
/// dispatch, returning the non-reduced axis's length (the dispatch width)
/// and the reduced axis's length (the scan width).
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= 2`,
/// `output_shape` does not equal `input_shape` with `axis` collapsed to one,
/// or the reduced axis is empty.
pub fn validate_arg_reduce_shape(
    axis: usize,
    input_shape: [usize; 2],
    output_shape: [usize; 2],
) -> Result<(usize, usize)> {
    if axis >= 2 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("arg-reduce axis {axis} is out of range for rank 2"),
        });
    }
    let other = 1 - axis;
    let mut expected = input_shape;
    expected[axis] = 1;
    if output_shape != expected {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "arg-reduce output shape mismatch: expected {expected:?} \
                 (input {input_shape:?} with axis {axis} collapsed), got {output_shape:?}"
            ),
        });
    }
    let axis_len = input_shape[axis];
    if axis_len == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "arg-reduce over an empty axis is undefined".to_string(),
        });
    }
    Ok((input_shape[other], axis_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_collapsed_shape_on_either_axis() {
        assert_eq!(
            validate_arg_reduce_shape(0, [3, 4], [1, 4]).expect("axis 0"),
            (4, 3)
        );
        assert_eq!(
            validate_arg_reduce_shape(1, [3, 4], [3, 1]).expect("axis 1"),
            (3, 4)
        );
    }

    #[test]
    fn rejects_an_out_of_range_axis() {
        let err = validate_arg_reduce_shape(2, [3, 4], [1, 4]).expect_err("axis 2 on rank 2");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_mismatched_output_shape() {
        let err = validate_arg_reduce_shape(0, [3, 4], [3, 4]).expect_err("axis not collapsed");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_reduced_axis() {
        let err = validate_arg_reduce_shape(0, [0, 4], [1, 4]).expect_err("empty axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
