//! Device-neutral n-D tiling over strided views.
//!
//! [`crate::TileOps`] writes `reps[d]` concatenated copies of a strided
//! input along every axis `d` into a caller-owned strided output:
//! `output[i] = input[i with each lane taken modulo the input extent]`.
//! Input and output share rank `N`; the output's extent along each axis is
//! the input's times its rep count. This is the `torch.Tensor.repeat`
//! semantic `coeus_ops::tile` documents, restricted to the fixed-rank
//! case: where the coeus entry point pads leading size-1 dims when `reps`
//! is longer or shorter than the rank, this seam takes exactly one rep
//! per axis and a rank change stays a reshape-plus-tile composition at
//! the caller. The device kernel reads/writes through the same
//! [`crate::StridedView`] every other accelerator seam accepts —
//! transposed, sliced, and offset views dispatch without a contiguous
//! staging copy.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Per-axis repetition counts, one per operation axis.
pub type TileReps<const N: usize> = [usize; N];

/// Device-neutral n-D tiling.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `L: TileOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input and output share rank `N`. `output.layout.shape()[d]` must equal
/// `input.layout.shape()[d] * reps[d]` for every axis `d`. A zero rep
/// count empties its axis, legally: tiling zero times yields zero cells.
/// The products are checked — an overflowing rep count is a shape error,
/// never a wrapped extent.
///
/// # Special values
///
/// Tiling is a pure copy fan-out; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait TileOps<D: ComputeDevice, T: Pod> {
    /// Write `reps[d]` concatenated copies of `input` along every axis.
    ///
    /// The output buffer must not alias the input buffer: every thread
    /// reads the input while other threads write the output, so sharing
    /// storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when an output extent is not its
    /// input extent times its rep count, a rep count overflows, the
    /// output buffer aliases the input buffer, a layout is unsupported,
    /// or the backend dispatch fails.
    fn tile_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        reps: TileReps<N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate that `output_shape` is `input_shape` tiled by `reps`.
///
/// The axes check in order from `0`, so a caller bug reports
/// deterministically regardless of backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when any output
/// dimension does not equal `input_shape[d] * reps[d]`, or when that
/// product overflows `usize`.
pub fn validate_tile_shape<const N: usize>(
    input_shape: [usize; N],
    reps: TileReps<N>,
    output_shape: [usize; N],
) -> Result<()> {
    for d in 0..N {
        let expected = input_shape[d].checked_mul(reps[d]).ok_or_else(|| {
            HephaestusError::InvalidConfiguration {
                message: format!(
                    "tile rep count overflows axis {d} extent: {} * {}",
                    input_shape[d], reps[d]
                ),
            }
        })?;
        if output_shape[d] != expected {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "tile output shape mismatch at axis {d}: expected {expected} \
                     (input {} * reps {}), got {}",
                    input_shape[d], reps[d], output_shape[d]
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
    fn accepts_the_exact_tile_geometry() {
        validate_tile_shape([2, 2], [2, 3], [4, 6]).expect("valid geometry");
    }

    #[test]
    fn accepts_all_ones_reps() {
        validate_tile_shape([2, 3], [1, 1], [2, 3]).expect("identity reps are valid");
    }

    #[test]
    fn accepts_a_zero_rep() {
        validate_tile_shape([2, 3], [2, 0], [4, 0]).expect("zero reps are valid");
    }

    #[test]
    fn rejects_an_output_extent_mismatch() {
        let err = validate_tile_shape([2, 2], [2, 3], [4, 5]).expect_err("extent mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_overflowing_rep() {
        let err =
            validate_tile_shape([2, 3], [usize::MAX, 1], [0, 3]).expect_err("overflow must fail");
        assert!(
            matches!(&err, HephaestusError::InvalidConfiguration { message } if message
                .contains("overflows")),
            "overflow must report as such, got {err:?}"
        );
    }
}
