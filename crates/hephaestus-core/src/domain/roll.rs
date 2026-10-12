//! Device-neutral rank-2 axis roll (circular shift).
//!
//! Part of the shape/indexing family (`HEPH-SHAPE-OPS-PROVIDER-1`), split
//! out the way `TopKOps`/`InterpolationOps`/`AdaptivePoolingOps` were: one
//! self-contained algorithm at a time rather than one seam covering fifteen
//! unrelated operations. Matches the family's established rank-2/axis
//! convention, but `output`'s shape equals `input`'s exactly (a roll never
//! resizes an axis) — the shift wraps circularly rather than remapping to a
//! new extent.
//!
//! `output[i] = input[(i - shift).rem_euclid(axis_len)]` for every position
//! `i` along `axis` — `shift` may be negative (rolls the other direction)
//! or exceed `axis_len` in magnitude (wraps multiple times); `rem_euclid`
//! keeps the result in `[0, axis_len)` for every sign and magnitude, so no
//! special-casing is needed at the boundary.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::view::StridedView;

/// Device-neutral rank-2 axis roll (circular shift).
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `R: RollOps<D, T>` costs nothing at runtime and every call monomorphizes
/// to the backend's own kernel dispatch.
pub trait RollOps<D: ComputeDevice, T: Pod> {
    /// Circularly shift each lane of `input` along `axis` by `shift`
    /// positions into `output`, whose shape equals `input`'s exactly.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis >= 2`, `input`'s reduced-
    /// axis length is zero, `output`'s shape does not equal `input`'s, or
    /// the backend dispatch fails.
    fn roll_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        shift: i64,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>;
}

/// The source index a roll of `shift` positions (over an axis of length
/// `axis_len`) reads for destination index `dst_idx`.
///
/// # Panics
///
/// Panics when `axis_len` is zero; callers validate that boundary before
/// dispatching.
#[must_use]
pub fn roll_source_index(dst_idx: usize, shift: i64, axis_len: usize) -> usize {
    assert!(axis_len != 0, "invariant: roll axis length is non-zero");
    let modulus = axis_len as u128;
    let destination = (dst_idx as u128) % modulus;
    let magnitude = (shift.unsigned_abs() as u128) % modulus;
    if shift.is_negative() {
        ((destination + magnitude) % modulus) as usize
    } else {
        ((destination + modulus - magnitude) % modulus) as usize
    }
}

/// Validate `axis` and that `input_shape`/`output_shape` match exactly for
/// a roll dispatch, returning `(lanes, axis_len)`.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= 2`,
/// the reduced axis is empty, or `output_shape != input_shape`.
pub fn validate_roll_shape(
    axis: usize,
    input_shape: [usize; 2],
    output_shape: [usize; 2],
) -> Result<(usize, usize)> {
    if axis >= 2 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("roll axis {axis} is out of range for rank 2"),
        });
    }
    if input_shape != output_shape {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "roll output shape mismatch: expected {input_shape:?}, got {output_shape:?}"
            ),
        });
    }
    let axis_len = input_shape[axis];
    if axis_len == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "roll over an empty axis is undefined".to_string(),
        });
    }
    Ok((input_shape[1 - axis], axis_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_shift_wraps_forward() {
        // axis_len=5, shift=2: dst 0 reads src (0-2).rem_euclid(5)=3.
        assert_eq!(roll_source_index(0, 2, 5), 3);
        assert_eq!(roll_source_index(1, 2, 5), 4);
        assert_eq!(roll_source_index(2, 2, 5), 0);
    }

    #[test]
    fn negative_shift_wraps_backward() {
        assert_eq!(roll_source_index(0, -1, 5), 1);
        assert_eq!(roll_source_index(4, -1, 5), 0);
    }

    #[test]
    fn shift_beyond_axis_len_wraps_multiple_times() {
        // shift=7 on axis_len=5 is equivalent to shift=2.
        assert_eq!(roll_source_index(0, 7, 5), roll_source_index(0, 2, 5));
        assert_eq!(roll_source_index(0, -7, 5), roll_source_index(0, -2, 5));
    }

    #[test]
    fn signed_extreme_shifts_do_not_overflow() {
        assert_eq!(
            roll_source_index(1, i64::MIN, 5),
            roll_source_index(1, i64::MIN.rem_euclid(5), 5)
        );
        assert_eq!(
            roll_source_index(4, i64::MAX, 5),
            roll_source_index(4, i64::MAX.rem_euclid(5), 5)
        );
    }

    #[test]
    fn zero_shift_is_identity() {
        for i in 0..5 {
            assert_eq!(roll_source_index(i, 0, 5), i);
        }
    }

    #[test]
    fn accepts_matching_shapes_on_either_axis() {
        assert_eq!(
            validate_roll_shape(0, [3, 4], [3, 4]).expect("axis 0"),
            (4, 3)
        );
        assert_eq!(
            validate_roll_shape(1, [3, 4], [3, 4]).expect("axis 1"),
            (3, 4)
        );
    }

    #[test]
    fn rejects_an_out_of_range_axis() {
        let err = validate_roll_shape(2, [3, 4], [3, 4]).expect_err("axis 2 on rank 2");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_resized_output() {
        let err = validate_roll_shape(0, [3, 4], [5, 4]).expect_err("resized axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_axis() {
        let err = validate_roll_shape(0, [0, 4], [0, 4]).expect_err("empty axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
