//! Device-neutral n-D circular shift over strided views.
//!
//! [`crate::RollOps`] writes a circularly shifted copy of a strided input
//! into a caller-owned strided output: for each `(shift, dim)` pair,
//! `output[i] = input[i with the `dim` lane moved back by `shift` modulo
//! the extent]`, wrapping past either boundary. Input and output share
//! rank `N` and shape exactly. This is the `torch.roll` semantic
//! `coeus_ops::roll` documents, while the device kernel reads/writes
//! through the same [`crate::StridedView`] every other accelerator seam
//! accepts — transposed, sliced, and offset views dispatch without a
//! contiguous staging copy.
//!
//! Shifts pair with dims positionally, exactly as the coeus entry point
//! takes them: `shifts[k]` applies along `dims[k]`. Rolls along distinct
//! dims commute and rolls along one dim add, so backends may fold every
//! pair into one effective per-axis shift rather than applying pairs
//! sequentially — duplicate dims compose by summation, and the result is
//! identical either way.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D circular shift.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `R: RollOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input and output share rank `N`, and both shapes must be equal: a roll
/// permutes cells, never resizes. `shifts` and `dims` must have equal
/// length, and every dim must be below `N`.
///
/// # Shifts
///
/// Each shift is a signed displacement in cells along its dim: positive
/// moves elements toward higher indices with the tail wrapping to the
/// front, negative the reverse. Shifts of any magnitude are legal —
/// backends reduce them modulo the dim extent — including shifts larger
/// than the extent and shifts along size-1 dims (which are identities).
/// An empty shift list is the identity.
///
/// # Special values
///
/// A roll is a pure permutation; `T` is never interpreted, so no
/// special-value handling applies beyond what the scalar type itself
/// represents.
pub trait RollOps<D: ComputeDevice, T: Pod> {
    /// Write `input` circularly shifted by `shifts[k]` along `dims[k]`.
    ///
    /// The output buffer must not alias the input buffer: every thread
    /// reads the input while other threads write the output, so sharing
    /// storage would race.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `shifts` and `dims` disagree
    /// in length, a dim is out of range, the output shape is not the
    /// input shape, the output buffer aliases the input buffer, a layout
    /// is unsupported, or the backend dispatch fails.
    fn roll_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        shifts: &[isize],
        dims: &[usize],
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the roll geometry: shift/dim pairing, dim range, and
/// input/output shape equality.
///
/// The checks run in a fixed order — pairing length, dim range, shape
/// equality — so a caller bug reports deterministically regardless of
/// backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `shifts` and
/// `dims` disagree in length, a dim is `>= N`, or the output shape is not
/// the input shape.
pub fn validate_roll_shape<const N: usize>(
    input_shape: [usize; N],
    shifts: &[isize],
    dims: &[usize],
    output_shape: [usize; N],
) -> Result<()> {
    if shifts.len() != dims.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "roll shifts/dims length mismatch: {} shifts against {} dims",
                shifts.len(),
                dims.len()
            ),
        });
    }
    for &dim in dims {
        if dim >= N {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("roll dim {dim} is out of bounds for rank {N}"),
            });
        }
    }
    for (d, (&input_extent, &output_extent)) in
        input_shape.iter().zip(output_shape.iter()).enumerate()
    {
        if input_extent != output_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "roll input/output shape mismatch at axis {d}: expected {input_extent}, got {output_extent}",
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
        validate_roll_shape([2, 4], &[1], &[1], [2, 4]).expect("valid geometry");
    }

    #[test]
    fn accepts_an_empty_shift_list() {
        validate_roll_shape([2, 4], &[], &[], [2, 4]).expect("identity roll is valid");
    }

    #[test]
    fn accepts_duplicate_dims() {
        // Duplicate dims compose by summation; they are not an error.
        validate_roll_shape([2, 4], &[1, -1], &[1, 1], [2, 4]).expect("duplicate dims are valid");
    }

    #[test]
    fn rejects_a_shift_dim_length_mismatch() {
        let err = validate_roll_shape([2, 4], &[1, 2], &[1], [2, 4]).expect_err("length mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_dim_out_of_range() {
        let err = validate_roll_shape([2, 4], &[1], &[2], [2, 4]).expect_err("dim out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_mismatch() {
        let err = validate_roll_shape([2, 4], &[1], &[1], [2, 3]).expect_err("output mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
