//! Device-neutral n-D conditional selection over strided views.
//!
//! [`crate::WhereOps`] writes the element-wise ternary `output[i] =
//! on_true[i]` wherever the condition cell is nonzero and `on_false[i]`
//! elsewhere, into a caller-owned strided output. Condition and both
//! branches share rank `N` and shape exactly — the same-shape contract
//! `coeus_ops::where_cond` documents, which today composes the selection
//! from seven backend-primitive dispatches with intermediate temporaries
//! while this seam runs it as one kernel. The device kernel reads/writes
//! through the same [`crate::StridedView`] every other accelerator seam
//! accepts — transposed, sliced, and offset views dispatch without a
//! contiguous staging copy.
//!
//! The condition carries the same scalar `T` as the branches: a cell
//! selects its `on_true` value exactly when it compares unequal to zero.
//! Like [`crate::MaskedFillOps`], this seam interprets `T` rather than
//! copying it, so it names its kernel dialect for the non-finite edges.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::dialect::KernelDialect;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D conditional selection.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `W: WhereOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Condition, both branches, and output share rank `N`, and all four
/// shapes must be equal. There is no broadcasting: any extent that
/// disagrees with the condition is a shape error, exactly as
/// `coeus_ops::where_cond` rejects it.
///
/// # Condition test
///
/// A cell selects `on_true` exactly when its condition value compares
/// unequal to zero under the scalar's equality. For finite values that
/// pins every edge on every backend: any nonzero magnitude selects
/// `on_true`, and negative zero selects `on_false` (`-0.0 == 0.0`).
/// Backends must not reinterpret a finite condition bitwise — a naive
/// bitwise test would select `on_true` under `-0.0` — nor threshold it.
///
/// # Special values
///
/// Selected cells copy their branch bitwise. A `NaN` condition selects
/// `on_true` only where the kernel dialect advertises IEEE special
/// values (see [`KernelDialect::IEEE_SPECIAL_VALUES`], ADR 0043):
/// unordered NaN comparison is exactly what a non-advertising dialect
/// like WGSL is permitted to get wrong, so no assertion about a `NaN`
/// condition can hold there. Consumers needing cross-backend
/// determinism must keep conditions finite unless every dialect involved
/// advertises the guarantee.
pub trait WhereOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors conditional-selection kernels in.
    type Dialect: KernelDialect;

    /// Write `on_true` under nonzero condition cells, `on_false` elsewhere.
    ///
    /// The output buffer must alias none of the three source buffers:
    /// every thread reads all three sources while other threads write the
    /// output, so sharing storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the four shapes disagree, the
    /// output buffer aliases a source buffer, a layout is unsupported, or
    /// the backend dispatch fails.
    fn where_into<const N: usize>(
        &self,
        device: &D,
        cond: StridedView<'_, D::Buffer<T>, N>,
        on_true: StridedView<'_, D::Buffer<T>, N>,
        on_false: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the conditional-selection geometry: all four shapes must be
/// equal.
///
/// The checks run in a fixed order — `on_true`, `on_false`, then output,
/// each against the condition — so a caller bug reports deterministically
/// regardless of backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when any branch or
/// output shape disagrees with the condition shape on any axis.
pub fn validate_where_shape<const N: usize>(
    cond_shape: [usize; N],
    on_true_shape: [usize; N],
    on_false_shape: [usize; N],
    output_shape: [usize; N],
) -> Result<()> {
    for (d, (&cond_extent, &branch_extent)) in
        cond_shape.iter().zip(on_true_shape.iter()).enumerate()
    {
        if cond_extent != branch_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "where cond/on_true shape mismatch at axis {d}: expected {cond_extent}, got {branch_extent}",
                ),
            });
        }
    }
    for (d, (&cond_extent, &branch_extent)) in
        cond_shape.iter().zip(on_false_shape.iter()).enumerate()
    {
        if cond_extent != branch_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "where cond/on_false shape mismatch at axis {d}: expected {cond_extent}, got {branch_extent}",
                ),
            });
        }
    }
    for (d, (&cond_extent, &output_extent)) in
        cond_shape.iter().zip(output_shape.iter()).enumerate()
    {
        if cond_extent != output_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "where cond/output shape mismatch at axis {d}: expected {cond_extent}, got {output_extent}",
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
        validate_where_shape([2, 4], [2, 4], [2, 4], [2, 4]).expect("valid geometry");
    }

    #[test]
    fn accepts_an_empty_extent() {
        validate_where_shape([2, 0], [2, 0], [2, 0], [2, 0]).expect("empty is valid");
    }

    #[test]
    fn rejects_an_on_true_shape_mismatch() {
        let err =
            validate_where_shape([2, 4], [2, 3], [2, 4], [2, 4]).expect_err("on_true mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_on_false_shape_mismatch() {
        let err =
            validate_where_shape([2, 4], [2, 4], [2, 3], [2, 4]).expect_err("on_false mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_mismatch() {
        let err =
            validate_where_shape([2, 4], [2, 4], [2, 4], [2, 3]).expect_err("output mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn reports_mismatches_in_branch_order() {
        // Fixed check order: every shape disagrees, the on_true error wins.
        let err = validate_where_shape([2, 4], [2, 3], [2, 5], [2, 6]).expect_err("all mismatch");
        assert!(
            matches!(&err, HephaestusError::InvalidConfiguration { message } if message.contains("cond/on_true")),
            "on_true mismatch must report first, got {err:?}"
        );
    }
}
