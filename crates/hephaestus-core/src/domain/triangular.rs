//! Device-neutral rank-2 triangular masking (tril / triu).
//!
//! Part of the shape/indexing family (`HEPH-SHAPE-OPS-PROVIDER-1`), with one
//! self-contained operation per seam. Unlike the family's other rank-2 seams, this one has no `axis`
//! parameter — it masks the whole 2D view by each element's `(row, col)`
//! position relative to a diagonal, not by scanning one axis.
//!
//! `output[row, col] = input[row, col]` when the element is on the kept
//! side of the diagonal offset by `diagonal`, else `T::ZERO`.
//! [`TriangularMode::Lower`](crate::domain::triangular::TriangularMode::Lower)
//! keeps `col <= row + diagonal` (matching
//! `numpy.tril`'s convention: `diagonal = 0` is the main diagonal,
//! positive shifts it up-right, negative down-left);
//! [`TriangularMode::Upper`](crate::domain::triangular::TriangularMode::Upper)
//! keeps `col >= row + diagonal`
//! (`numpy.triu`'s convention, the mirror condition).

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::view::StridedView;

/// Which side of the diagonal survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangularMode {
    /// Keep `col <= row + diagonal` (numpy's `tril`).
    Lower,
    /// Keep `col >= row + diagonal` (numpy's `triu`).
    Upper,
}

/// Device-neutral rank-2 triangular masking.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `M: TriangularOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait TriangularOps<D: ComputeDevice, T: Pod> {
    /// Mask `input` by `mode` and `diagonal` into `output`, whose shape
    /// equals `input`'s exactly.
    ///
    /// Backends validate storage bounds, reject overlapping output views and
    /// reject input/output aliasing before dispatch. The core contract accepts
    /// the full signed `i64` diagonal range. WGPU carries element offsets and
    /// strides as signed `i32` shader values and clamps the diagonal threshold
    /// to that coordinate-difference range before upload. This preserves the
    /// full `i64` comparison semantics while rejecting addresses outside the
    /// signed shader range. WGPU accepts scalar storage layouts whose element
    /// stride matches WGSL and performs no conversion copy.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `output`'s shape does not equal
    /// `input`'s, or the backend dispatch fails.
    fn triangular_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        mode: TriangularMode,
        diagonal: i64,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>;
}

/// Whether element `(row, col)` survives `mode`'s mask at the given
/// `diagonal` offset.
#[must_use]
pub fn triangular_keeps(mode: TriangularMode, row: usize, col: usize, diagonal: i64) -> bool {
    let difference = if col >= row {
        (
            true,
            u128::try_from(col - row).expect("invariant: usize fits u128"),
        )
    } else {
        (
            false,
            u128::try_from(row - col).expect("invariant: usize fits u128"),
        )
    };
    let diagonal_magnitude = u128::from(diagonal.unsigned_abs());
    match mode {
        TriangularMode::Lower => match difference {
            (true, magnitude) => {
                diagonal >= 0
                    && magnitude
                        <= u128::try_from(diagonal).expect("invariant: nonnegative diagonal")
            }
            (false, magnitude) => diagonal >= 0 || magnitude >= diagonal_magnitude,
        },
        TriangularMode::Upper => match difference {
            (true, magnitude) => {
                diagonal < 0
                    || magnitude
                        >= u128::try_from(diagonal).expect("invariant: nonnegative diagonal")
            }
            (false, magnitude) => diagonal < 0 && magnitude <= diagonal_magnitude,
        },
    }
}

/// Validate that `input_shape` and `output_shape` match exactly for a
/// triangular-masking dispatch.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when
/// `output_shape != input_shape`.
pub fn validate_triangular_shape(input_shape: [usize; 2], output_shape: [usize; 2]) -> Result<()> {
    if input_shape != output_shape {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "triangular output shape mismatch: expected {input_shape:?}, got {output_shape:?}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_keeps_the_main_diagonal_and_below_at_zero_offset() {
        assert!(triangular_keeps(TriangularMode::Lower, 0, 0, 0));
        assert!(triangular_keeps(TriangularMode::Lower, 1, 0, 0));
        assert!(!triangular_keeps(TriangularMode::Lower, 0, 1, 0));
    }

    #[test]
    fn upper_keeps_the_main_diagonal_and_above_at_zero_offset() {
        assert!(triangular_keeps(TriangularMode::Upper, 0, 0, 0));
        assert!(triangular_keeps(TriangularMode::Upper, 0, 1, 0));
        assert!(!triangular_keeps(TriangularMode::Upper, 1, 0, 0));
    }

    #[test]
    fn positive_diagonal_shifts_the_boundary_up_right() {
        // diagonal=1: lower keeps col <= row+1, so (0,1) is now kept.
        assert!(triangular_keeps(TriangularMode::Lower, 0, 1, 1));
        assert!(!triangular_keeps(TriangularMode::Lower, 0, 2, 1));
    }

    #[test]
    fn negative_diagonal_shifts_the_boundary_down_left() {
        // diagonal=-1: lower keeps col <= row-1, so (0,0) is now excluded.
        assert!(!triangular_keeps(TriangularMode::Lower, 0, 0, -1));
        assert!(triangular_keeps(TriangularMode::Lower, 1, 0, -1));
    }

    #[test]
    fn lower_and_upper_are_complementary_off_diagonal() {
        for row in 0..4usize {
            for col in 0..4usize {
                if col != row {
                    assert_ne!(
                        triangular_keeps(TriangularMode::Lower, row, col, 0),
                        triangular_keeps(TriangularMode::Upper, row, col, 0),
                        "row={row} col={col}"
                    );
                }
            }
        }
    }

    #[test]
    fn diagonal_extremes_keep_the_mathematical_boundary() {
        assert!(triangular_keeps(TriangularMode::Lower, 0, 0, i64::MAX));
        assert!(!triangular_keeps(TriangularMode::Upper, 0, 0, i64::MAX));
        assert!(!triangular_keeps(TriangularMode::Lower, 0, 0, i64::MIN));
        assert!(triangular_keeps(TriangularMode::Upper, 0, 0, i64::MIN));
        assert!(triangular_keeps(
            TriangularMode::Lower,
            usize::MAX,
            0,
            i64::MIN + 1
        ));
        assert!(!triangular_keeps(TriangularMode::Upper, 1, 0, i64::MAX - 1));
    }

    #[test]
    fn signed_difference_handles_usize_extremes_without_clamping() {
        assert!(!triangular_keeps(
            TriangularMode::Lower,
            0,
            usize::MAX,
            i64::MAX
        ));
        assert!(triangular_keeps(
            TriangularMode::Upper,
            0,
            usize::MAX,
            i64::MAX
        ));
        assert!(triangular_keeps(
            TriangularMode::Lower,
            usize::MAX,
            0,
            i64::MIN
        ));
        assert!(!triangular_keeps(
            TriangularMode::Upper,
            usize::MAX,
            0,
            i64::MIN
        ));
    }

    #[test]
    fn predicate_matches_widened_mathematical_comparison() {
        for row in 0..=4 {
            for col in 0..=4 {
                for diagonal in [i64::MIN, -5, -1, 0, 1, 5, i64::MAX] {
                    let boundary = i128::try_from(row).expect("invariant: usize fits i128")
                        + i128::from(diagonal);
                    let coordinate = i128::try_from(col).expect("invariant: usize fits i128");
                    assert_eq!(
                        triangular_keeps(TriangularMode::Lower, row, col, diagonal),
                        coordinate <= boundary,
                        "lower row={row} col={col} diagonal={diagonal}"
                    );
                    assert_eq!(
                        triangular_keeps(TriangularMode::Upper, row, col, diagonal),
                        coordinate >= boundary,
                        "upper row={row} col={col} diagonal={diagonal}"
                    );
                }
            }
        }
    }

    #[test]
    fn accepts_matching_shapes() {
        validate_triangular_shape([3, 4], [3, 4]).expect("matching shapes");
    }

    #[test]
    fn rejects_a_mismatched_shape() {
        let err = validate_triangular_shape([3, 4], [4, 3]).expect_err("mismatched shapes");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
