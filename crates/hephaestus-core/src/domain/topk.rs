//! Device-neutral rank-2 top-`k` selection.
//!
//! Companion to the argmax/argmin seam (`HEPH-TOPK-ARGMINMAX-PROVIDER-1`):
//! that seam returns the single extremum's index, this one returns the `k`
//! largest elements along `axis` — both value and index, sorted descending —
//! with the reduced axis resized to `k` rather than collapsed to one. `k` is
//! validated against the input's static shape before dispatch: unlike an
//! index buffer's contents, `axis_len` is shape metadata the host already
//! has, so this check needs no device-side flag (contrast the embedding
//! gather seam's indices, which are device-resident data and cannot be
//! validated before dispatch).
//!
//! Ties break toward the lowest index: a later element strictly greater than
//! a current top-`k` member displaces it, but an equal value never does.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::view::StridedView;

/// Device-neutral rank-2 top-`k` selection.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `K: TopKOps<D, T>` costs nothing at runtime and every call monomorphizes
/// to the backend's own kernel dispatch.
pub trait TopKOps<D: ComputeDevice, T: Pod + PartialOrd> {
    /// Write the `k` largest elements of each lane of `input` along `axis`
    /// into `values` (descending) and their source indices into `indices`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis >= 2`, `k == 0`, `k` exceeds
    /// the reduced axis's length, `values`/`indices` do not share the shape
    /// of `input` with `axis` resized to `k`, or the backend dispatch fails.
    fn topk_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        k: usize,
        values: StridedView<'_, D::Buffer<T>, 2>,
        indices: StridedView<'_, D::Buffer<u32>, 2>,
    ) -> Result<()>;
}

/// Validate `axis`, `k`, `input_shape`, and the two output shapes for a
/// top-`k` dispatch, returning `(lanes, axis_len)`.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= 2`,
/// `k == 0`, `k` exceeds the reduced axis's length, or either output shape
/// does not equal `input_shape` with `axis` resized to `k`.
pub fn validate_topk_shape(
    axis: usize,
    k: usize,
    input_shape: [usize; 2],
    values_shape: [usize; 2],
    indices_shape: [usize; 2],
) -> Result<(usize, usize)> {
    if axis >= 2 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("top-k axis {axis} is out of range for rank 2"),
        });
    }
    let other = 1 - axis;
    let axis_len = input_shape[axis];
    if k == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "top-k k must be at least 1".to_string(),
        });
    }
    if k > axis_len {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("top-k k={k} exceeds the reduced axis length {axis_len}"),
        });
    }
    let mut expected = input_shape;
    expected[axis] = k;
    if values_shape != expected {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "top-k values shape mismatch: expected {expected:?}, got {values_shape:?}"
            ),
        });
    }
    if indices_shape != expected {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "top-k indices shape mismatch: expected {expected:?}, got {indices_shape:?}"
            ),
        });
    }
    Ok((input_shape[other], axis_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_valid_k_and_shapes() {
        assert_eq!(
            validate_topk_shape(1, 2, [3, 4], [3, 2], [3, 2]).expect("valid"),
            (3, 4)
        );
    }

    #[test]
    fn rejects_zero_k() {
        let err = validate_topk_shape(1, 0, [3, 4], [3, 0], [3, 0]).expect_err("k=0");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_k_larger_than_the_axis() {
        let err = validate_topk_shape(1, 5, [3, 4], [3, 5], [3, 5]).expect_err("k>axis_len");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_mismatched_values_shape() {
        let err = validate_topk_shape(1, 2, [3, 4], [3, 3], [3, 2]).expect_err("bad values shape");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
