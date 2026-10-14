//! Device-neutral inner product over strided views.
//!
//! [`crate::DotOps`] folds the elementwise products of two strided inputs
//! into one caller-owned scalar cell: `sum(a[i] * b[i])` over the inputs'
//! shared logical flat order. This is the `torch.dot` semantic
//! `coeus_ops::dot` documents — any two shapes with equal numel, empty
//! yielding zero — except the scalar crosses explicitly in a `[1]` cell
//! rather than returning to the host, the nonzero count-cell precedent:
//! returning `T` would force a synchronization point into every dispatch,
//! while the explicit cell keeps the seam's uniform `Result<()>` shape
//! and leaves the caller in control of when the scalar synchronizes. The
//! device kernel reads through the same [`crate::StridedView`] every
//! other accelerator seam accepts — transposed, sliced, and offset
//! views dispatch without a contiguous staging copy.
//!
//! Accumulation is a single pass in native `T` precision with no
//! widening accumulator: products form and add in ascending flat order
//! starting from zero, so every backend agrees bitwise on every input
//! including non-finite ones. The kernel is pure arithmetic — no
//! comparison touches a value — so unlike the ordering and masking
//! seams it needs no [`KernelDialect`](crate::KernelDialect)
//! branch: there is no unordered-compare hazard to opt out of.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral inner product.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `O: DotOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// The inputs may have different ranks `N` and `M` and different shapes
/// — only their cell counts must agree, checked as validated products.
/// The output is rank 1 with shape `[1]`.
///
/// # Order
///
/// Products `a[i] * b[i]` form over the shared logical flat order
/// (`i` ascending from `0` to `numel`) and accumulate into zero in that
/// same order. Backends implement this as one single-pass fold — no
/// tree, no tiles, no atomics — so the summation association is identical
/// on every backend and float results agree bitwise, not within a
/// tolerance. An empty input yields zero without reading either input.
///
/// # Aliasing
///
/// The output buffer must not alias either input buffer: the single
/// write races the input reads it folds. The two inputs may freely
/// alias each other — `dot(x, x)` is legal and common — since both are
/// only read.
pub trait DotOps<D: ComputeDevice, T: Pod> {
    /// Fold the elementwise products of `a` and `b` into `out`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the input numels disagree or
    /// overflow, the output shape is not `[1]`, the output buffer
    /// aliases either input buffer, a layout is unsupported, or the
    /// backend dispatch fails.
    fn dot_into<const N: usize, const M: usize>(
        &self,
        device: &D,
        a: StridedView<'_, D::Buffer<T>, N>,
        b: StridedView<'_, D::Buffer<T>, M>,
        out: StridedView<'_, D::Buffer<T>, 1>,
    ) -> Result<()>;
}

/// Validate the dot geometry: both numels against each other and the
/// output shape against `[1]`.
///
/// The checks run in a fixed order — `a` overflow, `b` overflow, numel
/// mismatch, output length — so a caller bug reports deterministically
/// regardless of backend. The numel bounds are checked cell-count
/// products, which reject an overflowing shape before any backend sizes
/// a dispatch from it.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when either input
/// shape's product overflows, the numels disagree, or the output shape
/// is not `[1]`.
pub fn validate_dot_shape<const N: usize, const M: usize>(
    a_shape: [usize; N],
    b_shape: [usize; M],
    out_shape: [usize; 1],
) -> Result<()> {
    let mut a_numel = 1usize;
    for extent in a_shape {
        a_numel =
            a_numel
                .checked_mul(extent)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: format!("dot input shape {a_shape:?} overflows numel"),
                })?;
    }
    let mut b_numel = 1usize;
    for extent in b_shape {
        b_numel =
            b_numel
                .checked_mul(extent)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: format!("dot input shape {b_shape:?} overflows numel"),
                })?;
    }
    if a_numel != b_numel {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("dot numel mismatch: {a_numel} against {b_numel}"),
        });
    }
    if out_shape[0] != 1 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("dot output length must be 1, got {}", out_shape[0]),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_matching_numels_of_different_rank() {
        validate_dot_shape([2, 4], [8], [1]).expect("valid geometry");
    }

    #[test]
    fn accepts_empty_inputs() {
        validate_dot_shape([2, 0], [0], [1]).expect("empty is valid");
    }

    #[test]
    fn rejects_a_numel_mismatch() {
        let err = validate_dot_shape([2, 4], [7], [1]).expect_err("numel mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_non_scalar_output() {
        let err = validate_dot_shape([2, 4], [8], [2]).expect_err("non-scalar output");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_overflowing_shape() {
        let err = validate_dot_shape([usize::MAX, 2], [8], [1]).expect_err("overflowing shape");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn reports_the_mismatch_before_the_output_length() {
        // Fixed check order: both disagree, the mismatch error wins.
        let err = validate_dot_shape([2, 4], [7], [2]).expect_err("both disagree");
        assert!(
            matches!(&err, HephaestusError::InvalidConfiguration { message } if message.contains("numel mismatch")),
            "mismatch must report first, got {err:?}"
        );
    }
}
