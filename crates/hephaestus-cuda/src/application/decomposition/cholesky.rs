//! GPU-resident Cholesky decomposition.
//!
//! Computes **A** = **L** **L**ᵀ for symmetric positive-definite matrices.
//! Delegates to the CPU via [`leto_ops`], storing the result on the device and
//! as a host-side decomposition for solve/inv/det without re-factorization.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use super::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Lower-triangular Cholesky factor on the device, with host-side
/// decomposition for solve/inv/det without re-factorization.
pub struct GpuCholesky {
    /// Host-side leto-ops decomposition (owns the factor data).
    inner: leto_ops::CholeskyDecomposition<f32>,
    /// Device-resident lower-triangular factor **L** (*n* × *n*, row-major).
    lower: CudaBuffer<f32>,
    n: usize,
}

impl GpuCholesky {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lower(&self) -> &CudaBuffer<f32> {
        &self.lower
    }

    /// Consume and return the lower-triangular factor buffer.
    #[must_use]
    #[inline]
    pub fn into_lower(self) -> CudaBuffer<f32> {
        self.lower
    }

    /// Determinant det(**A**) = Πᵢ Lᵢᵢ² via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side forward/back substitution.
    ///
    /// Downloads the RHS from the device, solves on the host using the
    /// stored decomposition, and uploads the solution vector.
    pub fn solve(&self, device: &CudaDevice, rhs: &CudaBuffer<f32>) -> Result<CudaBuffer<f32>> {
        if rhs.len() != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len(),
            });
        }
        if self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0f32; self.n];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([self.n]).unwrap(),
            &rhs_host,
        );
        let x = self
            .inner
            .solve(&rhs_view)
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky solve failed: {e}"),
            })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }

    /// Compute the inverse **A**⁻¹ via the host-side decomposition.
    pub fn inv(&self, device: &CudaDevice) -> Result<CudaBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inv = self
            .inner
            .inv()
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

/// Compute the Cholesky factorization **A** = **L** **L**ᵀ on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`]. The result is stored on the device for downstream GPU
/// consumers.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let lower = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::cholesky_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([0, 0]).unwrap(),
            &[],
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Cholesky decomposition failed: {e}"),
        })?;
        return Ok(GpuCholesky { inner, lower, n: 0 });
    }

    // Download input to host.
    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    // Create a leto ArrayView over the downloaded data.
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    // Factor on CPU using leto-ops.
    let chol =
        leto_ops::cholesky_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Cholesky decomposition failed: {e}"),
        })?;

    // Upload the lower-triangular factor to the device.
    let lower = device.upload(leto::Storage::as_slice(chol.lower().storage()))?;

    Ok(GpuCholesky {
        inner: chol,
        lower,
        n,
    })
}
