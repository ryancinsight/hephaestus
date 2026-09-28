//! Device-neutral rank-2 triangular masking (tril / triu).
//!
//! Part of the shape/indexing family (`HEPH-SHAPE-OPS-PROVIDER-1`), split
//! out the way `RollOps` was (ADR 0069): one self-contained algorithm at a
//! time. Unlike the family's other rank-2 seams, this one has no `axis`
//! parameter — it masks the whole 2D view by each element's `(row, col)`
//! position relative to a diagonal, not by scanning one axis.
//!
//! `output[row, col] = input[row, col]` when the element is on the kept
//! side of the diagonal offset by `diagonal`, else `T::ZERO`.
//! [`TriangularMode::Lower`] keeps `col <= row + diagonal` (matching
//! `numpy.tril`'s convention: `diagonal = 0` is the main diagonal,
//! positive shifts it up-right, negative down-left);
//! [`TriangularMode::Upper`] keeps `col >= row + diagonal`
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
    let row = i64::try_from(row).unwrap_or(i64::MAX);
    let col = i64::try_from(col).unwrap_or(i64::MAX);
    let boundary = row.saturating_add(diagonal);
    match mode {
        TriangularMode::Lower => col <= boundary,
        TriangularMode::Upper => col >= boundary,
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
    fn accepts_matching_shapes() {
        validate_triangular_shape([3, 4], [3, 4]).expect("matching shapes");
    }

    #[test]
    fn rejects_a_mismatched_shape() {
        let err = validate_triangular_shape([3, 4], [4, 3]).expect_err("mismatched shapes");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
