//! Rank, determinant, pseudoinverse, and matrix exponential.

use super::scalars::MatrixRankScalar;
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Layout;

/// Host-side partial-pivoting Gaussian elimination, computing numerical rank
/// and (for square inputs) determinant in one pass over `scratch`.
///
/// Port of the row-reduction algorithm the WGSL `hephaestus-matrix-properties`
/// kernel used to run in a single `@workgroup_size(1)` invocation (WG-P1):
/// that kernel gained nothing from GPU dispatch — one thread does O(rows·cols²)
/// scalar work while every other lane in the launch sits idle — and paid the
/// full pipeline/dispatch/readback overhead for it. This host port preserves
/// the exact pivot-selection and elimination order (same threshold
/// `max_abs * relative_tolerance`, same sign-flip-on-swap determinant
/// tracking), so the two pinned contract tests that assert this algorithm's
/// specific divergence from Leto's SVD-spectrum criterion
/// (`matrix_rank_relative_tolerance_is_the_discriminator`,
/// `det_of_near_singular_triangular_is_exact_pivot_product`) still hold —
/// this is a dispatch-mechanism change, not an algorithm change.
pub(crate) fn matrix_properties_host<T>(
    scratch: &mut [T],
    rows: usize,
    cols: usize,
    tolerance: f32,
) -> (usize, T)
where
    T: MatrixRankScalar,
{
    let zero = T::from(0.0);
    let one = T::from(1.0);
    let abs = |v: T| if v < zero { -v } else { v };

    if rows == 0 || cols == 0 {
        return (0, zero);
    }

    let square = rows == cols;
    let mut max_abs = zero;
    for &value in scratch.iter() {
        let magnitude = abs(value);
        if magnitude > max_abs {
            max_abs = magnitude;
        }
    }

    if max_abs <= zero {
        return (0, zero);
    }

    let threshold = max_abs * T::from(tolerance);
    let mut rank = 0usize;
    let mut det = one;
    let mut sign = one;

    for col in 0..cols {
        if rank >= rows {
            break;
        }

        let mut pivot_row = rank;
        let mut pivot_abs = zero;
        for row in rank..rows {
            let magnitude = abs(scratch[row * cols + col]);
            if magnitude > pivot_abs {
                pivot_abs = magnitude;
                pivot_row = row;
            }
        }

        if pivot_abs > threshold {
            if pivot_row != rank {
                sign = -sign;
                for swap_col in 0..cols {
                    scratch.swap(rank * cols + swap_col, pivot_row * cols + swap_col);
                }
            }

            let pivot = scratch[rank * cols + col];
            if square {
                det = det * pivot;
            }
            for row in 0..rows {
                if row != rank {
                    let factor = scratch[row * cols + col] / pivot;
                    for elim_col in col..cols {
                        let target_idx = row * cols + elim_col;
                        let source = rank * cols + elim_col;
                        scratch[target_idx] = scratch[target_idx] - factor * scratch[source];
                    }
                }
            }
            rank += 1;
        }
    }

    let determinant = if square && rank == rows {
        sign * det
    } else {
        zero
    };
    (rank, determinant)
}

/// Runs a matrix operation over mutable canonical row-major storage.
///
/// Canonical C-contiguous layouts already occupy the downloaded buffer prefix
/// in the order required by the operation, so that prefix is reused in place.
/// Other valid layouts are compacted into an independently owned buffer.
pub(crate) fn with_row_major_matrix<T, R>(
    host_buffer: &mut [T],
    layout: &Layout<2>,
    operation: impl FnOnce(&mut [T]) -> R,
) -> Result<R>
where
    T: Copy,
{
    layout
        .validate_storage_len(host_buffer.len())
        .map_err(map_layout_err)?;
    let logical_len = layout.checked_size().map_err(map_layout_err)?;

    if layout.is_c_contiguous() {
        let storage_len = host_buffer.len();
        let scratch = host_buffer.get_mut(..logical_len).ok_or_else(|| {
            HephaestusError::DispatchFailed {
                message: format!(
                    "contiguous matrix requires {logical_len} elements but backing storage has {}",
                    storage_len
                ),
            }
        })?;
        return Ok(operation(scratch));
    }

    let [rows, cols] = layout.shape();
    let mut scratch = Vec::with_capacity(logical_len);
    for row in 0..rows {
        for col in 0..cols {
            let source_offset = layout.offset_of([row, col]).map_err(map_layout_err)?;
            let value = host_buffer.get(source_offset).copied().ok_or_else(|| {
                HephaestusError::DispatchFailed {
                    message: format!(
                        "matrix element [{row}, {col}] resolves to offset {source_offset}, outside backing storage length {}",
                        host_buffer.len()
                    ),
                }
            })?;
            scratch.push(value);
        }
    }

    Ok(operation(&mut scratch))
}

pub(crate) fn matrix_properties_with_tolerance<T>(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, T, 2>,
    relative_tolerance: f32,
) -> Result<(usize, WgpuBuffer<T>)>
where
    T: MatrixRankScalar,
{
    let [rows, cols] = matrix.layout.shape();
    if rows == 0 || cols == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matrix rank is undefined for empty matrix with shape {:?}",
                matrix.layout.shape()
            ),
        });
    }
    if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matrix rank tolerance must be finite and non-negative, got {relative_tolerance}"
            ),
        });
    }

    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    // Host-delegated (WG-P1): the single-thread `@workgroup_size(1)` kernel
    // this replaced gained nothing from GPU dispatch. Download the whole
    // backing buffer once. Canonical row-major inputs reuse that allocation;
    // strided views compact into row-major scratch for in-place elimination.
    let mut host_buffer = vec![T::from(0.0); matrix.buffer.len];
    device.download(matrix.buffer, &mut host_buffer)?;

    let (rank, determinant) = with_row_major_matrix(&mut host_buffer, matrix.layout, |scratch| {
        matrix_properties_host(scratch, rows, cols, relative_tolerance)
    })?;

    let det_out = device.upload(&[determinant])?;
    Ok((rank, det_out))
}

/// Estimate the numerical rank of a finite rank-2 matrix on the GPU.
///
/// The kernel performs Gaussian row reduction in GPU storage memory and counts
/// pivots greater than `relative_tolerance * max(abs(matrix))`. This matches
/// Leto's relative-threshold intent for exact finite test cases, but it is a
/// row-reduction criterion rather than Leto's SVD-spectrum criterion.
///
/// # Ill-conditioned inputs
///
/// `relative_tolerance * max(abs(matrix))` is the sole rank discriminator: a
/// pivot at or below it is treated as zero. Because this is a pivot-magnitude
/// criterion while Leto's is an SVD-spectrum criterion, the two can disagree on
/// ill-conditioned matrices where pivot magnitudes and singular values differ;
/// they agree when those coincide (diagonal / orthogonally-scaled inputs). The
/// threshold-boundary behaviour is pinned by the contract test
/// `matrix_rank_relative_tolerance_is_the_discriminator`.
pub fn matrix_rank_with_tolerance<T>(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, T, 2>,
    relative_tolerance: f32,
) -> Result<usize>
where
    T: MatrixRankScalar,
{
    matrix_properties_with_tolerance(device, matrix, relative_tolerance).map(|(rank, _)| rank)
}

/// Estimate the numerical rank of a finite rank-2 matrix on the GPU.
///
/// Uses Leto's default relative tolerance of `1e-9`.
#[inline]
pub fn matrix_rank<T>(device: &WgpuDevice, matrix: StridedOperand<'_, T, 2>) -> Result<usize>
where
    T: MatrixRankScalar,
{
    matrix_rank_with_tolerance(device, matrix, 1.0e-9)
}

/// Compute the determinant of a finite square matrix on the GPU.
///
/// The kernel performs Gaussian row reduction in GPU storage memory and returns
/// zero for singular matrices, matching Leto's determinant contract for exact
/// finite cases.
///
/// # Ill-conditioned inputs
///
/// `det` applies no determinant tolerance (it passes `relative_tolerance == 0`):
/// only an exactly-zero pivot collapses the determinant to zero, so a
/// near-singular matrix returns its small but nonzero pivot product rather than
/// a tolerance-zeroed `0`. This can diverge from an SVD- or tolerance-based
/// determinant on ill-conditioned inputs. Pinned by the contract test
/// `det_of_near_singular_triangular_is_exact_pivot_product`.
pub fn det<T>(device: &WgpuDevice, matrix: StridedOperand<'_, T, 2>) -> Result<WgpuBuffer<T>>
where
    T: MatrixRankScalar,
{
    let [rows, cols] = matrix.layout.shape();
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "det requires a square matrix, got shape {:?}",
                matrix.layout.shape()
            ),
        });
    }
    matrix_properties_with_tolerance(device, matrix, 0.0).map(|(_, determinant)| determinant)
}

/// Compute the Moore-Penrose pseudoinverse A⁺ of a device-resident matrix.
///
/// Host-delegated: the matrix is downloaded, the pseudoinverse computed on
/// the CPU via `leto_ops::pinv` (SVD-based), and the result uploaded. This
/// provides API parity over device buffers, not a GPU kernel; the transfer
/// is O(rows·cols) each way.
#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub fn pinv(device: &WgpuDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<WgpuBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape();
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    if rows == 0 || cols == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let out_arr = leto_ops::pinv(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Pseudoinverse failed: {e}"),
    })?;

    device.upload(leto::Storage::as_slice(out_arr.storage()))
}

/// Compute the matrix exponential e^A of a device-resident matrix.
///
/// Host-delegated: the matrix is downloaded, e^A computed on the CPU via
/// `leto_ops::matexp` (scaling-and-squaring), and the result uploaded. This
/// provides API parity over device buffers, not a GPU kernel; the transfer
/// is O(n²) each way.
#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub fn matexp(device: &WgpuDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<WgpuBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape();
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Matrix exponential requires square matrix, got shape [{rows}, {cols}]"
            ),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    if rows == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let out_arr = leto_ops::matexp(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Matrix exponential failed: {e}"),
    })?;

    device.upload(leto::Storage::as_slice(out_arr.storage()))
}
