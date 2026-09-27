//! Device-neutral n-D padding over strided views.
//!
//! [`crate::PadOps`] writes a padded copy of a strided input into a
//! caller-owned strided output: cells inside the original region copy the
//! source, cells in the pad margins take a caller-supplied fill value. This
//! mirrors `leto::pad`'s `[(before, after); N]` contract so host and device
//! results are directly comparable, while the device kernel reads/writes
//! through the same [`crate::StridedView`] every other accelerator seam
//! accepts — transposed, sliced, and offset views dispatch without a
//! contiguous staging copy.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Per-axis `(before, after)` padding widths, matching leto's `PadWidth<N>`.
pub type PadWidth<const N: usize> = [(usize, usize); N];

/// Device-neutral n-D padding.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `P: PadOps<D, T>` costs nothing at runtime and every call monomorphizes to
/// the backend's own kernel dispatch.
///
/// # Shape
///
/// `output.layout.shape()[d]` must equal
/// `width[d].0 + input.layout.shape()[d] + width[d].1` for every axis `d`.
///
/// # Special values
///
/// `fill` is written verbatim into every pad-margin cell; no special-value
/// handling applies beyond what the scalar type itself represents.
pub trait PadOps<D: ComputeDevice, T: Pod> {
    /// Write a padded copy of `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `output`'s shape does not match
    /// the padded shape implied by `width`, a layout is unsupported, or the
    /// backend dispatch fails.
    fn pad_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        width: PadWidth<N>,
        fill: T,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate that `output_shape` is `input_shape` padded by `width`, returning
/// the packed `(pad_before, input_shape)` pair the kernel metadata needs.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when any output
/// dimension does not equal `width[d].0 + input_shape[d] + width[d].1`.
pub fn validate_pad_shape<const N: usize>(
    input_shape: [usize; N],
    width: PadWidth<N>,
    output_shape: [usize; N],
) -> Result<()> {
    for d in 0..N {
        let (before, after) = width[d];
        let expected = before
            .checked_add(input_shape[d])
            .and_then(|v| v.checked_add(after))
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: format!("pad width overflows axis {d} extent"),
            })?;
        if output_shape[d] != expected {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "pad output shape mismatch at axis {d}: expected {expected} \
                     (before {before} + input {} + after {after}), got {}",
                    input_shape[d], output_shape[d]
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
    fn accepts_the_exact_padded_shape() {
        assert!(validate_pad_shape([3, 4], [(1, 2), (0, 1)], [6, 5]).is_ok());
    }

    #[test]
    fn accepts_zero_padding_as_identity_shape() {
        assert!(validate_pad_shape([3, 4], [(0, 0), (0, 0)], [3, 4]).is_ok());
    }

    #[test]
    fn rejects_an_output_shape_missing_the_pad_margin() {
        let err =
            validate_pad_shape([3, 4], [(1, 2), (0, 1)], [3, 4]).expect_err("missing pad margin");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_overlarge_output_shape() {
        let err = validate_pad_shape([3, 4], [(1, 2), (0, 1)], [7, 5])
            .expect_err("overlarge output shape");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
