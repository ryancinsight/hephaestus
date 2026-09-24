//! GPU-resident QR decomposition via Householder reflectors.
//!
//! Computes **A** = **Q R** where **Q** is orthogonal and **R** is
//! upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`qr_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`qr_decompose_blocked`] — blocked algorithm where panel
//!   factorization runs on the CPU but the trailing Householder application
//!   runs on the GPU via a dedicated CUDA kernel.
//!
//! # Mathematical Foundations
//!
//! ## Blocked QR with GPU Trailing Application
//!
//! For large *m*, the dominant cost is applying the *b* Householder
//! reflectors from each panel to the trailing submatrix.  Each application
//! costs O(m(n−k)) flops and is embarrassingly parallel across columns.
//!
//! **Theorem (Blocked QR complexity).** For *m × n* with block size *b*,
//! the total flop count is 2n²(m − n/3), identical to unblocked QR. ∎

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

#[cfg(feature = "cuda")]
use hephaestus_core::{BlockedDecompositionBackend, BlockedQrBackend, TrailingHh, blocked_qr};

#[cfg(feature = "cuda")]
use super::validate::validate_dense_operand;

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

#[cfg(feature = "cuda")]
mod householder;
#[cfg(feature = "cuda")]
mod q;

/// QR decomposition result: device-resident R factor with host-side
/// decomposition for solve_least_squares.
pub struct GpuQrDecomposition {
    /// Host-side leto-ops decomposition (owns packed/heads/betas).
    inner: leto_ops::QrDecomposition<f32>,
    /// Device-resident upper-triangular factor **R** (*m* × *n*, row-major).
    r: CudaBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuQrDecomposition {
    /// (rows, cols) of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r_buffer(&self) -> &CudaBuffer<f32> {
        &self.r
    }

    /// Take ownership of the device-resident **R** buffer.
    ///
    /// This avoids a device-to-device copy when a caller needs **R** as an
    /// independent result after consuming the decomposition.
    #[must_use]
    #[inline]
    pub fn into_r_buffer(self) -> CudaBuffer<f32> {
        self.r
    }

    /// Borrow the host-side Leto decomposition.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::QrDecomposition<f32> {
        &self.inner
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ (least squares).
    pub fn solve_least_squares(
        &self,
        device: &CudaDevice,
        rhs: &CudaBuffer<f32>,
    ) -> Result<CudaBuffer<f32>> {
        let (m, n) = (self.rows, self.cols);
        if rhs.len() != m {
            return Err(HephaestusError::LengthMismatch {
                host_len: m,
                device_len: rhs.len(),
            });
        }
        if m == 0 || n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let rhs_host = device.download_owned(rhs)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([m]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
        let x = self.inner.solve_least_squares(&rhs_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("QR least-squares solve failed: {e}"),
            }
        })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }
}

/// Compute the Householder QR factorization on the GPU.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-finite values in the input.
/// - Exactly-zero pivot column norm (rank-deficient input).
pub fn qr_decompose(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [rows, cols] = matrix.layout.shape();
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let host_data = device.download_owned(matrix.buffer)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let qr = leto_ops::qr_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("QR decomposition failed: {e}"),
    })?;

    let r_host = qr.r();
    let r_buf = device.upload(leto::Storage::as_slice(r_host.storage()))?;

    Ok(GpuQrDecomposition {
        inner: qr,
        r: r_buf,
        rows,
        cols,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing Householder application
// ---------------------------------------------------------------------------

/// Panel block size for the blocked QR algorithm.
#[cfg(feature = "cuda")]
const QR_BLOCK_SIZE: usize = 32;

/// Blocked QR factorization **A = Q R** with GPU-accelerated trailing
/// Householder application.
///
/// The operand must be dense C-contiguous at offset 0 (the blocked path
/// bulk-copies the matrix storage on the device); transposed, offset, or
/// broadcast views are rejected with a typed error — materialize them
/// first.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-dense (non-C-contiguous / offset / broadcast) operand.
/// - Non-finite values in the input.
/// - Rank-deficient input (zero column norm).
pub fn qr_decompose_blocked(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    #[cfg(feature = "cuda")]
    {
        let [m, n] = matrix.layout.shape();
        if m < n {
            return Err(HephaestusError::DispatchFailed {
                message: format!("QR requires m ≥ n, got shape [{m}, {n}]"),
            });
        }
        matrix
            .layout
            .validate_storage_len(matrix.buffer.len())
            .map_err(map_layout_err)?;
        validate_dense_operand("QR", &matrix)?;

        if m == 0 || n == 0 {
            let r_buf = device.alloc_zeroed::<f32>(0)?;
            let inner =
                leto_ops::QrDecomposition::from_raw_parts(Vec::new(), Vec::new(), Vec::new(), m, n);
            return Ok(GpuQrDecomposition {
                inner,
                r: r_buf,
                rows: m,
                cols: n,
            });
        }

        let work_buf = device.clone_device(matrix.buffer, m * n)?;
        let block_size = QR_BLOCK_SIZE.min(n);
        let result = blocked_qr(device, work_buf, m, n, block_size)?;

        let inner = leto_ops::QrDecomposition::from_raw_parts(
            result.packed,
            result.heads,
            result.betas,
            m,
            n,
        );

        Ok(GpuQrDecomposition {
            inner,
            r: result.r,
            rows: m,
            cols: n,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, matrix);
        Err(HephaestusError::AdapterUnavailable {
            message: "hephaestus-cuda built without the `cuda` feature".to_string(),
        })
    }
}

// Custom gather/scatter compute kernels removed in favor of generic MatrixRegion transfers.

/// The CUDA blocked-QR operations the shared [`blocked_qr`] loop drives
/// (ADR 0003).
///
/// Only the reflector-metadata buffer and the trailing Householder kernel are
/// CUDA-specific: the loop owns the panel iteration, the CPU panel
/// factorisation, the reflector packing, and the sub-diagonal zeroing.
#[cfg(feature = "cuda")]
impl BlockedQrBackend for CudaDevice {
    type Reflectors = CudaBuffer<householder::HhReflectorMeta>;

    fn alloc_reflectors(&self, len: usize) -> Result<Self::Reflectors> {
        self.alloc_uninitialized::<householder::HhReflectorMeta>(len)
    }

    fn write_flat(&self, buf: &Self::Buffer, data: &[f32]) -> Result<()> {
        self.write_sub_buffer(buf, 0, data)
    }

    fn householder_trailing(
        &self,
        vectors: &Self::Buffer,
        matrix: &Self::Buffer,
        reflectors: &Self::Reflectors,
        spec: TrailingHh<'_>,
    ) -> Result<()> {
        householder::hh_trailing_update(
            self,
            householder::HhTrailingUpdate {
                vectors,
                matrix,
                reflectors,
                panel_rows: spec.panel_rows,
                trail_cols: spec.trail_cols,
                matrix_cols: spec.matrix_cols,
                panel_start: spec.panel_start,
                vector_offsets: spec.vector_offsets,
                betas: spec.betas,
            },
        )
    }
}
