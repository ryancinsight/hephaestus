//! GPU-resident QR decomposition via Householder reflectors.
//!
//! Computes **A** = **Q R** where **Q** is orthogonal and **R** is
//! upper-triangular. Delegates to the CPU via [`leto_ops`], storing the
//! result on the device and as a host-side decomposition for
//! solve_least_squares without re-factorization.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

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

    /// Solve min ‖**A** · **x** − **rhs**‖₂ (least squares).
    ///
    /// Downloads the RHS from the device, solves on the host using the
    /// stored Householder reflectors, and uploads the solution vector.
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

        let mut rhs_host = vec![0.0f32; m];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view =
            leto::ArrayView::<f32, 1>::new(leto::Layout::c_contiguous([m]).unwrap(), &rhs_host);
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
    let [rows, cols] = matrix.layout.shape;
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 || cols == 0 {
        let r_buf = device.alloc_zeroed::<f32>(0)?;
        // leto-ops rejects empty matrices; construct a minimal valid view
        // for the inner decomposition (the factorization loop is a no-op).
        let placeholder = vec![0.0f32];
        let inner = leto_ops::qr_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([1, 1]).unwrap(),
            &placeholder,
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("QR decomposition failed: {e}"),
        })?;
        return Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
            rows,
            cols,
        });
    }

    // Download to host.
    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let qr = leto_ops::qr_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("QR decomposition failed: {e}"),
    })?;

    // Materialize R on device from leto-ops result.
    let r_host = qr.r();
    let r_buf = device.upload(leto::Storage::as_slice(r_host.storage()))?;

    Ok(GpuQrDecomposition {
        inner: qr,
        r: r_buf,
        rows,
        cols,
    })
}
