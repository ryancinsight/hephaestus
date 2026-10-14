//! Device-neutral n-D masked fill over strided views.
//!
//! [`crate::MaskedFillOps`] writes `output[i] = value` wherever the mask
//! cell is nonzero and copies `input[i]` elsewhere, into a caller-owned
//! strided output. Input, mask, and output share rank `N` and shape
//! exactly — the same-shape contract `coeus_ops::masked_fill` documents —
//! while the device kernel reads/writes through the same
//! [`crate::StridedView`] every other accelerator seam accepts —
//! transposed, sliced, and offset views dispatch without a contiguous
//! staging copy.
//!
//! The mask carries the same scalar `T` as the input, binding the coeus
//! entry point directly with no conversion: a mask cell fills exactly when
//! it compares unequal to zero. The fill value crosses to device kernels
//! the way [`crate::PadOps`]' fill does, as a caller-supplied scalar
//! rather than packed metadata.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::dialect::KernelDialect;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D masked fill.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `M: MaskedFillOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input, mask, and output share rank `N`, and all three shapes must be
/// equal. There is no broadcasting: a mask extent that disagrees with the
/// input is a shape error, exactly as `coeus_ops::masked_fill` panics on
/// it.
///
/// # Mask test
///
/// A cell fills exactly when its mask value compares unequal to zero
/// under the scalar's equality. For finite values that pins every edge on
/// every backend: any nonzero magnitude, however small or negative,
/// fills, and negative zero keeps the input (`-0.0 == 0.0`). Backends
/// must not reinterpret a finite mask bitwise — a naive bitwise test
/// would fill under `-0.0` — nor threshold it.
///
/// # Special values
///
/// `value` is written verbatim into every filled cell; kept cells copy
/// the input bitwise. A `NaN` mask fills only where the kernel dialect
/// advertises IEEE special values (see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`], ADR 0043): unordered NaN
/// comparison is exactly what a non-advertising dialect like WGSL is
/// permitted to get wrong, so no assertion about a `NaN` mask can hold
/// there. Consumers needing cross-backend determinism must keep masks
/// finite unless every dialect involved advertises the guarantee.
pub trait MaskedFillOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors masked-fill kernels in.
    type Dialect: KernelDialect;

    /// Write `value` under nonzero mask cells, copying `input` elsewhere.
    ///
    /// The output buffer must alias neither the input buffer nor the mask
    /// buffer: every thread reads both sources while other threads write
    /// the output, so sharing storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the three shapes disagree, the
    /// output buffer aliases a source buffer, a layout is unsupported, or
    /// the backend dispatch fails.
    fn masked_fill_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        mask: StridedView<'_, D::Buffer<T>, N>,
        value: T,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the masked-fill geometry: all three shapes must be equal.
///
/// The checks run in a fixed order — mask against input, then output
/// against input — so a caller bug reports deterministically regardless
/// of backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when the mask shape
/// or the output shape disagrees with the input shape on any axis.
pub fn validate_masked_fill_shape<const N: usize>(
    input_shape: [usize; N],
    mask_shape: [usize; N],
    output_shape: [usize; N],
) -> Result<()> {
    for (d, (&input_extent, &mask_extent)) in input_shape.iter().zip(mask_shape.iter()).enumerate()
    {
        if input_extent != mask_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "masked_fill input/mask shape mismatch at axis {d}: expected {input_extent}, got {mask_extent}",
                ),
            });
        }
    }
    for (d, (&input_extent, &output_extent)) in
        input_shape.iter().zip(output_shape.iter()).enumerate()
    {
        if input_extent != output_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "masked_fill input/output shape mismatch at axis {d}: expected {input_extent}, got {output_extent}",
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
        validate_masked_fill_shape([2, 4], [2, 4], [2, 4]).expect("valid geometry");
    }

    #[test]
    fn accepts_an_empty_extent() {
        validate_masked_fill_shape([2, 0], [2, 0], [2, 0]).expect("empty is valid");
    }

    #[test]
    fn rejects_a_mask_shape_mismatch() {
        let err = validate_masked_fill_shape([2, 4], [2, 3], [2, 4]).expect_err("mask mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_mismatch() {
        let err = validate_masked_fill_shape([2, 4], [2, 4], [2, 3]).expect_err("output mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn reports_the_mask_mismatch_before_the_output_mismatch() {
        // Fixed check order: both disagree, the mask error wins.
        let err = validate_masked_fill_shape([2, 4], [2, 3], [2, 5]).expect_err("both mismatch");
        assert!(
            matches!(&err, HephaestusError::InvalidConfiguration { message }
                if message.contains("input/mask")),
            "mask mismatch must report first, got {err:?}"
        );
    }
}
