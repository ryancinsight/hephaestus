//! Device-neutral n-D stable sort over strided views.
//!
//! [`crate::SortOps`] stably sorts a strided input along one axis into two
//! caller-owned strided outputs: the sorted values and the `u32` argsort
//! indices holding each element's original position along the axis. Input
//! and both outputs share rank `N` and shape exactly. This is the
//! `torch.sort` semantic `coeus_ops::sort` documents, except the indices
//! are `u32` rather than `T`-encoded: positions are positions, and the
//! seam's index width is a seam-level choice like every other index in
//! the family, never a caller decision. The device kernel reads/writes
//! through the same [`crate::StridedView`] every other accelerator seam
//! accepts — transposed, sliced, and offset views dispatch without a
//! contiguous staging copy.
//!
//! Stability is structural, not aspirational: backends order by
//! `(value, original position)`, a total order under which equal values
//! keep their input order in both directions. The ordering bound is
//! `PartialOrd`, the [`crate::TopKOps`] precedent — no new trait.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::dialect::KernelDialect;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D stable sort.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `S: SortOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input, values, and indices share rank `N`, and all three shapes must
/// be equal: a sort permutes cells along one axis, never resizes. `dim`
/// must be below `N`.
///
/// # Order
///
/// With `descending = false`, each axis slice sorts ascending; with
/// `true`, descending. Either way the sort is stable: equal values keep
/// their input order, observably through the argsort indices. Backends
/// implement stability by breaking value ties toward the lower original
/// position — the `(value, position)` total order — so no backend's
/// parallel schedule can reorder equals.
///
/// # Special values
///
/// `NaN` sorts as greater than every other value, including infinity:
/// last ascending, first descending, stable among NaNs. This is the
/// `torch.sort` rule, and it holds only where the kernel dialect
/// advertises IEEE special values (see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`], ADR 0043): unordered NaN
/// comparison is exactly what a non-advertising dialect like WGSL is
/// permitted to get wrong, so no assertion about `NaN` placement can
/// hold there. Infinities and signed zeros sort by `PartialOrd` on every
/// backend (`-0.0 == 0.0` keeps input order).
pub trait SortOps<D: ComputeDevice, T: Pod + PartialOrd> {
    /// Kernel dialect this backend authors sorting kernels in.
    type Dialect: KernelDialect;

    /// Stably sort `input` along `dim` into `values` and `indices`.
    ///
    /// The values buffer must not alias the input buffer, and the indices
    /// buffer — `u32` against the input's `T` — can never name the same
    /// storage. Concretely: every thread reads the input while other
    /// threads write the values, so sharing that storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `dim` is out of range, either
    /// output shape is not the input shape, the values buffer aliases the
    /// input buffer, a layout is unsupported, or the backend dispatch
    /// fails.
    fn sort_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        dim: usize,
        descending: bool,
        values: StridedView<'_, D::Buffer<T>, N>,
        indices: StridedView<'_, D::Buffer<u32>, N>,
    ) -> Result<()>;
}

/// Validate the sort geometry: `dim` range and both output shapes against
/// the input.
///
/// The checks run in a fixed order — `dim` range, values shape, indices
/// shape — so a caller bug reports deterministically regardless of
/// backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `dim >= N` or
/// either output shape disagrees with the input shape on any axis.
pub fn validate_sort_shape<const N: usize>(
    input_shape: [usize; N],
    dim: usize,
    values_shape: [usize; N],
    indices_shape: [usize; N],
) -> Result<()> {
    if dim >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("sort dim {dim} is out of bounds for rank {N}"),
        });
    }
    for (d, (&input_extent, &values_extent)) in
        input_shape.iter().zip(values_shape.iter()).enumerate()
    {
        if input_extent != values_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "sort input/values shape mismatch at axis {d}: expected {input_extent}, got {values_extent}",
                ),
            });
        }
    }
    for (d, (&input_extent, &indices_extent)) in
        input_shape.iter().zip(indices_shape.iter()).enumerate()
    {
        if input_extent != indices_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "sort input/indices shape mismatch at axis {d}: expected {input_extent}, got {indices_extent}",
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
    fn accepts_matching_shapes() {
        validate_sort_shape([2, 4], 1, [2, 4], [2, 4]).expect("valid geometry");
    }

    #[test]
    fn accepts_an_empty_extent() {
        validate_sort_shape([2, 0], 1, [2, 0], [2, 0]).expect("empty is valid");
    }

    #[test]
    fn rejects_a_dim_out_of_range() {
        let err = validate_sort_shape([2, 4], 2, [2, 4], [2, 4]).expect_err("dim out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_values_shape_mismatch() {
        let err = validate_sort_shape([2, 4], 1, [2, 3], [2, 4]).expect_err("values mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_indices_shape_mismatch() {
        let err = validate_sort_shape([2, 4], 1, [2, 4], [2, 3]).expect_err("indices mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn reports_the_values_mismatch_before_the_indices_mismatch() {
        // Fixed check order: both disagree, the values error wins.
        let err = validate_sort_shape([2, 4], 1, [2, 3], [2, 5]).expect_err("both mismatch");
        assert!(
            matches!(&err, HephaestusError::InvalidConfiguration { message } if message.contains("input/values")),
            "values mismatch must report first, got {err:?}"
        );
    }
}
