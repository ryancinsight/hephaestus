//! Device-neutral n-D nonzero-coordinate extraction over strided views.
//!
//! [`crate::NonzeroOps`] scans a strided input in logical row-major order
//! and writes the coordinates of its nonzero cells — one `u32` row per
//! nonzero cell, each row holding that cell's `N` logical coordinates —
//! into a caller-owned rank-2 coordinates buffer, plus the nonzero count
//! `N` into a caller-owned single-cell count buffer. This is the
//! `torch.nonzero` semantic `coeus_ops::nonzero` documents, except the
//! coordinates are `u32` rather than `T`-encoded and the count crosses
//! explicitly: `N` is data-dependent, so the caller sizes the
//! coordinates for the worst case (`[numel, N]`) and learns `N` from the
//! count cell, which keeps the seam's uniform `Result<()>` shape and
//! leaves the caller in control of the count's synchronization. The
//! device kernel reads through the same [`crate::StridedView`] every
//! other accelerator seam accepts — transposed, sliced, and offset
//! views dispatch without a contiguous staging copy.
//!
//! A cell is nonzero exactly when it compares unequal to zero under the
//! scalar's equality, which pins the float edges: `-0.0` is zero on
//! every backend, while `NaN` is nonzero only where the kernel dialect
//! advertises IEEE special values (see
//! [`KernelDialect::IEEE_SPECIAL_VALUES`](crate::KernelDialect::IEEE_SPECIAL_VALUES),
//! ADR 0043). The skip is
//! wholesale, the sort seam's precedent: compaction packs rows densely,
//! so one unordered `NaN` comparison shifts every later row on a
//! non-advertising dialect and no assertion about a `NaN`-mixed input
//! can hold there.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::dialect::KernelDialect;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D nonzero-coordinate extraction.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `Z: NonzeroOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// The input has rank `N`. The coordinates output is always rank 2 with
/// shape `[numel, N]`, where `numel` is the input's checked cell count:
/// every cell could be nonzero. The count output is rank 1 with shape
/// `[1]`. `N` is the input rank as an extent, so a rank-1 input yields
/// single-column coordinates.
///
/// # Order
///
/// Coordinate rows run in the input's logical row-major order — the
/// order a flat `0..numel` scan visits the logical indices — so the
/// rows are deterministic on every backend regardless of parallel
/// schedule. Backends implement this with an output-parallel scan (each
/// row rescans the input for its nonzero), the scatter seam's
/// determinism-over-atomics precedent: a global atomic slot claim would
/// race the row order.
///
/// # Tail rows
///
/// Only the first `N` coordinate rows are written; rows `N..numel` are
/// never touched. Callers size for the worst case and must not assume
/// anything about the tail — backends write no fill there.
pub trait NonzeroOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors nonzero kernels in.
    type Dialect: KernelDialect;

    /// Write the nonzero coordinates of `input` into `coords` and their
    /// count into `count`.
    ///
    /// Neither output buffer may alias the input buffer, and the two
    /// outputs — both `u32` — may not alias each other either:
    /// coordinate writes race input reads across threads, and the
    /// count cell shares the outputs' element type, so sharing that
    /// storage would race too.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the coordinates shape is not
    /// `[numel, N]`, the count shape is not `[1]`, any two of the three
    /// buffers alias, a layout is unsupported, or the backend dispatch
    /// fails.
    fn nonzero_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        coords: StridedView<'_, D::Buffer<u32>, 2>,
        count: StridedView<'_, D::Buffer<u32>, 1>,
    ) -> Result<()>;
}

/// Validate the nonzero geometry: the coordinates shape against
/// `[numel, N]` and the count shape against `[1]`.
///
/// The checks run in a fixed order — coordinates rows, coordinates
/// columns, count length — so a caller bug reports deterministically
/// regardless of backend. The `numel` bound is the checked cell-count
/// product, which rejects a shape whose product overflows before any
/// backend sizes a dispatch from it.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when the input
/// shape's product overflows, the coordinates shape is not
/// `[numel, N]`, or the count shape is not `[1]`.
pub fn validate_nonzero_shape<const N: usize>(
    input_shape: [usize; N],
    coords_shape: [usize; 2],
    count_shape: [usize; 1],
) -> Result<()> {
    let mut numel = 1usize;
    for extent in input_shape {
        numel = numel
            .checked_mul(extent)
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: format!("nonzero input shape {input_shape:?} overflows numel"),
            })?;
    }
    if coords_shape[0] != numel {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "nonzero coords rows must be numel {numel}, got {}",
                coords_shape[0],
            ),
        });
    }
    if coords_shape[1] != N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "nonzero coords columns must be the input rank {N}, got {}",
                coords_shape[1]
            ),
        });
    }
    if count_shape[0] != 1 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("nonzero count length must be 1, got {}", count_shape[0]),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_worst_case_coords_and_scalar_count() {
        validate_nonzero_shape([2, 4], [8, 2], [1]).expect("valid geometry");
    }

    #[test]
    fn accepts_an_empty_input() {
        validate_nonzero_shape([2, 0], [0, 2], [1]).expect("empty is valid");
    }

    #[test]
    fn rejects_short_coords_rows() {
        let err = validate_nonzero_shape([2, 4], [7, 2], [1]).expect_err("short rows");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_wrong_coords_columns() {
        let err = validate_nonzero_shape([2, 4], [8, 3], [1]).expect_err("wrong columns");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_non_scalar_count() {
        let err = validate_nonzero_shape([2, 4], [8, 2], [2]).expect_err("non-scalar count");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn reports_the_rows_mismatch_before_the_columns_mismatch() {
        // Fixed check order: both disagree, the rows error wins.
        let err = validate_nonzero_shape([2, 4], [7, 3], [1]).expect_err("both mismatch");
        assert!(
            matches!(&err, HephaestusError::InvalidConfiguration { message } if message.contains("coords rows")),
            "rows mismatch must report first, got {err:?}"
        );
    }
}
