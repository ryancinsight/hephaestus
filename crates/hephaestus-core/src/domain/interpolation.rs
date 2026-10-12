//! Device-neutral rank-2 axis resampling (nearest / linear).
//!
//! Companion to [`crate::ArgReduceOps`] and [`crate::TopKOps`] in shape (a
//! rank-2 view resampled along one `axis`), but the resized axis's length is
//! caller-chosen rather than derived from the input (`output`'s shape need
//! not match `input`'s shape on `axis`) — this is a resize, not a reduction
//! or selection.
//!
//! Convention (pinned per `numerical_discipline`: convention pinning):
//! source coordinates map with the *align-corners* rule — output index `0`
//! always samples input index `0`, and (when `out_len > 1`) output index
//! `out_len - 1` always samples input index `in_len - 1`; interior points are
//! evenly spaced between. This is exact at both endpoints and well-defined
//! for `in_len == 1` (every output sample reads the sole input element) and
//! `out_len == 1` (the single output sample reads input index `0`). Weight
//! arithmetic runs entirely in `T`'s native precision via
//! `eunomia::TryFromCount::try_from_count` — no widen-compute-narrow cast (HARD per
//! `integrity`: fake generics).
//!
//! Bilinear (2D) resampling is not a separate mode: on a regular grid it is
//! separable into two 1D passes (resize width, then resize height), each an
//! ordinary call to `interpolate_axis_into` — documented in ADR 0067 rather
//! than encoded as a third, redundant kernel.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::view::StridedView;

/// Resampling kernel selected at the operation boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Sample the input element closest to the mapped source coordinate
    /// (ties round toward the lower index).
    Nearest,
    /// Linearly blend the two input elements bracketing the mapped source
    /// coordinate.
    Linear,
}

/// Device-neutral rank-2 axis resampling.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `I: InterpolationOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait InterpolationOps<D: ComputeDevice, T: Pod> {
    /// Resample each lane of `input` along `axis` into `output`, whose
    /// length on `axis` is the target size (independent of `input`'s).
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis >= 2`, `input`'s or
    /// `output`'s reduced-axis length is zero, `output`'s shape does not
    /// match `input`'s shape with `axis` resized, or the backend dispatch
    /// fails.
    fn interpolate_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        mode: InterpolationMode,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>;
}

/// Validate `axis`, `input_shape`, and `output_shape` for an interpolation
/// dispatch, returning `(lanes, in_len, out_len)`.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= 2`,
/// `in_len == 0`, `out_len == 0`, or `output_shape` does not equal
/// `input_shape` with `axis` resized to `out_len`.
pub fn validate_interpolation_shape(
    axis: usize,
    input_shape: [usize; 2],
    output_shape: [usize; 2],
) -> Result<(usize, usize, usize)> {
    if axis >= 2 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("interpolation axis {axis} is out of range for rank 2"),
        });
    }
    let other = 1 - axis;
    if input_shape[other] != output_shape[other] {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "interpolation output shape mismatch: expected the non-resized \
                 axis to stay {} (input {input_shape:?}), got {output_shape:?}",
                input_shape[other]
            ),
        });
    }
    let in_len = input_shape[axis];
    let out_len = output_shape[axis];
    if in_len == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "interpolation from an empty axis is undefined".to_string(),
        });
    }
    if out_len == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "interpolation to an empty axis is undefined".to_string(),
        });
    }
    Ok((input_shape[other], in_len, out_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_resized_axis_on_either_side() {
        assert_eq!(
            validate_interpolation_shape(0, [3, 4], [7, 4]).expect("axis 0"),
            (4, 3, 7)
        );
        assert_eq!(
            validate_interpolation_shape(1, [3, 4], [3, 9]).expect("axis 1"),
            (3, 4, 9)
        );
    }

    #[test]
    fn rejects_an_out_of_range_axis() {
        let err = validate_interpolation_shape(2, [3, 4], [3, 9]).expect_err("axis 2 on rank 2");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_mismatched_lane_count() {
        let err = validate_interpolation_shape(1, [3, 4], [5, 9]).expect_err("lane axis changed");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_input_axis() {
        let err = validate_interpolation_shape(0, [0, 4], [7, 4]).expect_err("empty input axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_output_axis() {
        let err = validate_interpolation_shape(0, [3, 4], [0, 4]).expect_err("empty output axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
