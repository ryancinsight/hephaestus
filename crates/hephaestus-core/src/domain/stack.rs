//! Device-neutral n-D stacking over strided views.
//!
//! [`crate::StackOps`] writes the stack of equal-shaped strided inputs along
//! a new `axis` into a caller-owned strided output of rank `M = N + 1`: the
//! output's `axis` extent is the input count, and every other output
//! dimension matches the inputs' shared shape. This mirrors `leto::stack`'s
//! contract so host and device results are directly comparable, while the
//! device kernel reads/writes through the same [`crate::StridedView`] every
//! other accelerator seam accepts — transposed, sliced, and offset views
//! dispatch without a contiguous staging copy.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D stacking.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `S: StackOps<D, T>` costs nothing at runtime and every call monomorphizes
/// to the backend's own kernel dispatch.
///
/// # Shape
///
/// Every input must share the same shape. `axis` ranges over `0..=N` (a new
/// trailing axis is legal); `M` must equal `N + 1`. The output's `axis`
/// extent must equal the input count, and every other output dimension must
/// match the inputs' shared shape with the `axis` lane skipped.
///
/// # Special values
///
/// Stacking is a bitwise copy; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait StackOps<D: ComputeDevice, T: Pod> {
    /// Write the stack of `inputs` along the new `axis` into `output`.
    ///
    /// The output buffer must not alias any input buffer: every thread
    /// reads its input while other threads (and later passes) write the
    /// output, so sharing storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `M != N + 1`, `axis` exceeds `N`,
    /// `inputs` is empty, the inputs disagree on any dimension,
    /// `output`'s shape does not match the stacked shape, the output buffer
    /// aliases an input buffer, a layout is unsupported, or the backend
    /// dispatch fails.
    fn stack_into<const N: usize, const M: usize>(
        &self,
        device: &D,
        inputs: &[StridedView<'_, D::Buffer<T>, N>],
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, M>,
    ) -> Result<()>
    where
        leto::RankMarker<N>: leto::InsertAxis<N, LargerShape = [usize; M]>;
}

/// Validate the stacking geometry.
///
/// The checks run in a fixed order — output rank, `axis` range, non-empty
/// inputs, input shape agreement, output shape — so a caller bug reports
/// deterministically regardless of backend.
///
/// Unlike [`validate_cat_shape`](super::cat::validate_cat_shape), this
/// returns no per-input bases: input `i` always lands at `axis` lane `i`,
/// so backends place blocks from the enumeration index directly.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `M != N + 1`,
/// `axis > N`, `input_shapes` is empty, two inputs disagree on any
/// dimension, the output's `axis` extent is not the input count, or any
/// other output dimension misses the inputs' shared shape.
pub fn validate_stack_shape<const N: usize, const M: usize>(
    input_shapes: &[[usize; N]],
    axis: usize,
    output_shape: [usize; M],
) -> Result<()> {
    if M != N + 1 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("stack output rank {M} must be one more than input rank {N}"),
        });
    }
    if axis > N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("stack axis {axis} is out of bounds for rank {N} inputs"),
        });
    }
    let Some(first) = input_shapes.first() else {
        return Err(HephaestusError::InvalidConfiguration {
            message: "stack requires at least one input".to_string(),
        });
    };

    for (index, shape) in input_shapes.iter().enumerate().skip(1) {
        for d in 0..N {
            if shape[d] != first[d] {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!(
                        "stack input {index} shape mismatch at axis {d}: expected {}, got {}",
                        first[d], shape[d]
                    ),
                });
            }
        }
    }

    if output_shape[axis] != input_shapes.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "stack output axis extent {} is not the input count {}",
                output_shape[axis],
                input_shapes.len()
            ),
        });
    }
    for (d, &extent) in output_shape.iter().enumerate() {
        if d == axis {
            continue;
        }
        let in_lane = if d < axis { d } else { d - 1 };
        if extent != first[in_lane] {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "stack output shape mismatch at axis {d}: expected {}, got {extent}",
                    first[in_lane]
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
    fn accepts_the_exact_stacked_shape_on_a_middle_axis() {
        validate_stack_shape(&[[2, 3], [2, 3], [2, 3]], 1, [2, 3, 3])
            .expect("valid stack geometry");
    }

    #[test]
    fn accepts_a_new_trailing_axis() {
        validate_stack_shape(&[[2, 3], [2, 3]], 2, [2, 3, 2]).expect("trailing axis is valid");
    }

    #[test]
    fn accepts_a_new_leading_axis() {
        validate_stack_shape(&[[2, 3], [2, 3]], 0, [2, 2, 3]).expect("leading axis is valid");
    }

    #[test]
    fn accepts_a_single_input_as_a_length_one_axis() {
        validate_stack_shape(&[[2, 3]], 1, [2, 1, 3]).expect("single input is valid");
    }

    #[test]
    fn rejects_an_output_rank_that_is_not_one_more() {
        let err = validate_stack_shape(&[[2, 3]], 0, [1, 2, 3, 4]).expect_err("rank mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_axis_past_the_new_trailing_lane() {
        let err = validate_stack_shape(&[[2, 3]], 3, [2, 3, 1]).expect_err("axis out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_input_list() {
        let err = validate_stack_shape::<2, 3>(&[], 0, [0, 2, 3]).expect_err("empty input list");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_input_shape_mismatch() {
        let err =
            validate_stack_shape(&[[2, 3], [2, 4]], 1, [2, 2, 3]).expect_err("input mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_axis_extent_missing_the_input_count() {
        let err =
            validate_stack_shape(&[[2, 3], [2, 3]], 1, [2, 3, 3]).expect_err("wrong axis extent");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_missing_a_shared_dimension() {
        let err = validate_stack_shape(&[[2, 3], [2, 3]], 1, [2, 2, 4])
            .expect_err("wrong shared dimension");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
