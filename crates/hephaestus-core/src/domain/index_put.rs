//! Device-neutral n-D scatter assignment over strided views.
//!
//! [`crate::IndexPutOps`] starts from a base input and writes values into it
//! at positions selected along one axis: for each selection `s`,
//! `output[j] = values[k]`, where `j` is `i` with the `dim` lane
//! replaced by `index[s]` and `k` is `i` with the `dim` lane replaced
//! by `s`, either replacing the base cell or accumulating onto it.
//! All three `T` operands share rank
//! `N`; every non-`dim` dimension agrees across all three, the output
//! matches the input's shape exactly, and the values' `dim` extent is the
//! index length. This is the inverse of [`crate::IndexSelectOps`] and the
//! `torch.index_put` semantic `coeus_ops::index_put` documents — except the
//! coeus entry point fixes `dim` at `0` and round-trips through the host,
//! while this seam selects along any axis and stays on the device.
//!
//! Indices are `u32`, matching [`crate::EmbeddingOps`]: the index width is a
//! seam-level choice, not a caller decision, so every backend packs and
//! bounds-checks one width.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Device-neutral n-D scatter assignment.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `I: IndexPutOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Shape
///
/// Input, values, and output share rank `N`. The output shape must equal
/// the input shape exactly; every non-`dim` dimension of the values must
/// agree with the input; the values' `dim` extent must equal the index
/// length. The index rank is concrete (`1`), so no rank-relation bound is
/// needed.
///
/// # Replace versus accumulate
///
/// With `accumulate = false`, each selection overwrites its target cell.
/// With `accumulate = true`, each selection adds onto it, and backends
/// apply contributions in ascending selection order — the same
/// accumulation-order rule [`crate::ScatterOps`] keeps, so an `f32`
/// fixture discriminates backends that reorder reductions. Duplicate
/// indices in replace mode are left unpinned, exactly as
/// `torch.index_put` leaves them undefined: only the accumulate path
/// constrains duplicate behavior (a sum).
///
/// # Out-of-range indices
///
/// An index value `>= input.shape()[dim]` is a data-dependent fault, not a
/// shape error: the host rejects it while scattering, and device backends
/// report it through the same one-`u32` atomic error flag the embedding
/// seam uses, downloaded after dispatch. No backend clamps or wraps.
///
/// # Special values
///
/// Replacement is a bitwise copy; accumulation is `T` addition, so the
/// usual floating-point order-dependence applies and is pinned by the
/// ascending-order rule above.
pub trait IndexPutOps<D: ComputeDevice, T: Pod> {
    /// Scatter `values` into a copy of `input` at `index` along `dim`.
    ///
    /// The output buffer must alias neither the input buffer nor the
    /// values buffer: every thread reads both sources while other threads
    /// write the output, so sharing storage would race. (The index buffer
    /// holds `u32` against the output's `T`, so the two cannot name one
    /// buffer.)
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when `dim` is out of range, the
    /// output shape is not the input shape, a values dimension disagrees,
    /// the values' `dim` extent is not the index length, any index value
    /// is out of range for the input's `dim` extent, the output buffer
    /// aliases a source buffer, a layout is unsupported, or the backend
    /// dispatch fails.
    #[allow(clippy::too_many_arguments)]
    fn index_put_into<const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        dim: usize,
        index: StridedView<'_, D::Buffer<u32>, 1>,
        values: StridedView<'_, D::Buffer<T>, N>,
        accumulate: bool,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>;
}

/// Validate the scatter-assignment geometry: `dim` range, output/input
/// agreement, values agreement, and values `dim` extent against the index
/// length.
///
/// The checks run in a fixed order — `dim` range, output shape, values
/// shape — so a caller bug reports deterministically regardless of
/// backend.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `dim >= N`, the
/// output shape is not the input shape, a values/non-`dim` dimension
/// disagrees, or the values' `dim` extent is not the index length.
pub fn validate_index_put_shape<const N: usize>(
    input_shape: [usize; N],
    dim: usize,
    index_len: usize,
    values_shape: [usize; N],
    output_shape: [usize; N],
) -> Result<()> {
    if dim >= N {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("index_put dim {dim} is out of bounds for rank {N}"),
        });
    }
    for (d, (&input_extent, &output_extent)) in
        input_shape.iter().zip(output_shape.iter()).enumerate()
    {
        if input_extent != output_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "index_put input/output shape mismatch at axis {d}: expected {input_extent}, got {output_extent}",
                ),
            });
        }
    }
    for (d, (&input_extent, &values_extent)) in
        input_shape.iter().zip(values_shape.iter()).enumerate()
    {
        if d == dim {
            if values_extent != index_len {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!(
                        "index_put values dim extent {values_extent} is not the index length {index_len}",
                    ),
                });
            }
        } else if input_extent != values_extent {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "index_put input/values shape mismatch at axis {d}: expected {input_extent}, got {values_extent}",
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
    fn accepts_the_exact_index_put_geometry() {
        validate_index_put_shape([2, 4], 1, 3, [2, 3], [2, 4]).expect("valid geometry");
    }

    #[test]
    fn accepts_a_full_range_update() {
        validate_index_put_shape([2, 4], 1, 4, [2, 4], [2, 4]).expect("full range is valid");
    }

    #[test]
    fn accepts_a_repeated_wider_than_input_update() {
        // The index length counts selections, not positions: repeats are
        // legal, so it may exceed the input's `dim` extent freely.
        validate_index_put_shape([2, 4], 1, 9, [2, 9], [2, 4]).expect("wide update is valid");
    }

    #[test]
    fn accepts_an_empty_index() {
        validate_index_put_shape([2, 4], 1, 0, [2, 0], [2, 4]).expect("vacuous update is valid");
    }

    #[test]
    fn rejects_a_dim_out_of_range() {
        let err =
            validate_index_put_shape([2, 4], 2, 3, [2, 3], [2, 4]).expect_err("dim out of range");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_an_output_shape_mismatch() {
        let err =
            validate_index_put_shape([2, 4], 1, 3, [2, 3], [2, 5]).expect_err("output mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_values_non_dim_mismatch() {
        let err =
            validate_index_put_shape([2, 4], 1, 3, [5, 3], [2, 4]).expect_err("values mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }

    #[test]
    fn rejects_a_values_dim_extent_missing_the_index_length() {
        let err = validate_index_put_shape([2, 4], 1, 3, [2, 2], [2, 4])
            .expect_err("dim extent mismatch");
        assert!(matches!(err, HephaestusError::InvalidConfiguration { .. }));
    }
}
