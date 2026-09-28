//! Device-neutral rank-2 adaptive pooling (average / maximum).
//!
//! Companion to [`crate::PoolingOps`] in role (a windowed reduction) but a
//! different algorithm: `PoolingOps` derives its windows from a fixed
//! kernel/stride/padding (`leto::WindowParameters`), while this seam derives
//! each output cell's window from mapping an arbitrary input extent onto a
//! caller-chosen output extent directly — the `AdaptiveAvgPool`/
//! `AdaptiveMaxPool` algorithm. Matches [`crate::InterpolationOps`]'s
//! shape/axis convention (rank-2, one resized `axis`) rather than
//! `PoolingOps`'s const-generic spatial rank, since both are a per-axis
//! extent remap.
//!
//! Window bounds are exact integer arithmetic (no float, unlike
//! `InterpolationOps`'s coordinate mapping): output index `i` (of `out_len`)
//! owns the half-open window `[i * in_len / out_len, ((i + 1) * in_len +
//! out_len - 1) / out_len)` — floor division for the start, ceiling
//! division for the end — which is always non-empty for `in_len >= 1` and
//! `out_len >= 1`. A rectangular 2D window's average or maximum equals the
//! average/maximum of the per-axis 1D reductions (both are associative,
//! commutative reductions over an axis-aligned Cartesian-product window), so
//! 2D adaptive pooling is not a separate mode here either — it composes as
//! two calls, one per spatial axis, exactly as `InterpolationOps`'s ADR
//! documents for bilinear resampling.

use eunomia::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::view::StridedView;

/// Window reduction selected at the operation boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptivePoolingMode {
    /// Average the input elements in each output cell's window.
    Average,
    /// Select the maximum input element in each output cell's window.
    Maximum,
}

/// Device-neutral rank-2 adaptive pooling.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `A: AdaptivePoolingOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait AdaptivePoolingOps<D: ComputeDevice, T: Pod> {
    /// Pool each lane of `input` along `axis` into `output`, whose length on
    /// `axis` is the target output extent (independent of `input`'s).
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `axis >= 2`, `input`'s or
    /// `output`'s reduced-axis length is zero, `output`'s shape does not
    /// match `input`'s shape with `axis` resized, or the backend dispatch
    /// fails.
    fn adaptive_pool_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        mode: AdaptivePoolingMode,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>;
}

/// The half-open window `[start, end)` output index `out_idx` (of `out_len`)
/// owns over an input axis of length `in_len`.
///
/// # Panics
///
/// Never panics for `out_len >= 1`; the caller (validation) guarantees this.
#[must_use]
pub fn adaptive_window(out_idx: usize, out_len: usize, in_len: usize) -> (usize, usize) {
    let start = out_idx * in_len / out_len;
    let end = ((out_idx + 1) * in_len).div_ceil(out_len);
    (start, end)
}

/// Validate `axis`, `input_shape`, and `output_shape` for an adaptive-pooling
/// dispatch, returning `(lanes, in_len, out_len)`.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `axis >= 2`,
/// `in_len == 0`, `out_len == 0`, or `output_shape` does not equal
/// `input_shape` with `axis` resized to `out_len`.
pub fn validate_adaptive_pooling_shape(
    axis: usize,
    input_shape: [usize; 2],
    output_shape: [usize; 2],
) -> Result<(usize, usize, usize)> {
    if axis >= 2 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("adaptive-pooling axis {axis} is out of range for rank 2"),
        });
    }
    let other = 1 - axis;
    if input_shape[other] != output_shape[other] {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "adaptive-pooling output shape mismatch: expected the non-resized \
                 axis to stay {} (input {input_shape:?}), got {output_shape:?}",
                input_shape[other]
            ),
        });
    }
    let in_len = input_shape[axis];
    let out_len = output_shape[axis];
    if in_len == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "adaptive-pooling from an empty axis is undefined".to_string(),
        });
    }
    if out_len == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "adaptive-pooling to an empty axis is undefined".to_string(),
        });
    }
    Ok((input_shape[other], in_len, out_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_covers_the_whole_axis_and_can_overlap_on_a_non_exact_split() {
        // in_len=7, out_len=3: windows [0,3), [2,5), [4,7) — every window is
        // size 3 (ceil(7/3)), covering every input index at least once;
        // windows 0/1 and 1/2 each share index 2 and 4 respectively. This
        // matches the standard adaptive-pooling algorithm (PyTorch's
        // AdaptiveAvgPool/AdaptiveMaxPool): windows overlap whenever
        // `out_len` does not evenly divide `in_len`, rather than partition
        // it into unequal contiguous pieces.
        assert_eq!(adaptive_window(0, 3, 7), (0, 3));
        assert_eq!(adaptive_window(1, 3, 7), (2, 5));
        assert_eq!(adaptive_window(2, 3, 7), (4, 7));
    }

    #[test]
    fn a_single_output_window_spans_the_whole_input() {
        assert_eq!(adaptive_window(0, 1, 7), (0, 7));
    }

    #[test]
    fn upsampling_still_yields_nonempty_windows() {
        // in_len=2, out_len=5: every window must be non-empty even though
        // out_len > in_len.
        for i in 0..5 {
            let (start, end) = adaptive_window(i, 5, 2);
            assert!(end > start, "window {i} was empty: [{start}, {end})");
        }
    }

    #[test]
    fn accepts_a_resized_axis_on_either_side() {
        assert_eq!(
            validate_adaptive_pooling_shape(0, [7, 4], [3, 4]).expect("axis 0"),
            (4, 7, 3)
        );
        assert_eq!(
            validate_adaptive_pooling_shape(1, [3, 7], [3, 4]).expect("axis 1"),
            (3, 7, 4)
        );
    }

    #[test]
    fn rejects_an_out_of_range_axis() {
        let err = validate_adaptive_pooling_shape(2, [3, 4], [3, 2]).expect_err("axis 2 on rank 2");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_mismatched_lane_count() {
        let err =
            validate_adaptive_pooling_shape(1, [3, 7], [5, 4]).expect_err("lane axis changed");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_input_axis() {
        let err = validate_adaptive_pooling_shape(0, [0, 4], [3, 4]).expect_err("empty input axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_empty_output_axis() {
        let err =
            validate_adaptive_pooling_shape(0, [7, 4], [0, 4]).expect_err("empty output axis");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
