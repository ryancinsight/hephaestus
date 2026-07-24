//! ROCm eigendecomposition surfaces backed by the shared Leto provider.
//!
//! Symmetric Jacobi eigenpairs and general complex eigenvalues mirror the
//! CUDA and wgpu result contracts. ROCm uploads the provider results into
//! typed device buffers and does not select a CPU or WGPU fallback.

use eunomia::Complex;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// Symmetric eigenpair result with device-resident values and vectors.
pub struct GpuSymmetricEigenDecomposition {
    inner: leto_ops::SymmetricEigenDecomposition<f32>,
    eigenvalues: RocmBuffer<f32>,
    eigenvectors: RocmBuffer<f32>,
    n: usize,
}

impl GpuSymmetricEigenDecomposition {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the eigenvalues buffer on the device.
    #[must_use]
    #[inline]
    pub fn eigenvalues(&self) -> &RocmBuffer<f32> {
        &self.eigenvalues
    }

    /// Borrow the eigenvectors buffer on the device.
    #[must_use]
    #[inline]
    pub fn eigenvectors(&self) -> &RocmBuffer<f32> {
        &self.eigenvectors
    }

    /// Borrow the shared Leto eigenpair result.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::SymmetricEigenDecomposition<f32> {
        &self.inner
    }
}

/// Compute symmetric Jacobi eigenpairs through the shared Leto provider.
pub fn symmetric_eigen_jacobi(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSymmetricEigenDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let eigenvalues = device.alloc_zeroed::<f32>(0)?;
        let eigenvectors = device.alloc_zeroed::<f32>(0)?;
        let layout = leto::Layout::c_contiguous([0, 0]).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("empty eigendecomposition layout failed: {error}"),
            }
        })?;
        let view = leto::ArrayView::<f32, 2>::new(layout, &[]);
        let inner = leto_ops::symmetric_eigen_jacobi(&view).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("empty eigendecomposition failed: {error}"),
            }
        })?;
        return Ok(GpuSymmetricEigenDecomposition {
            inner,
            eigenvalues,
            eigenvectors,
            n,
        });
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::symmetric_eigen_jacobi(&view).map_err(|error| {
        HephaestusError::DispatchFailed {
            message: format!("symmetric eigendecomposition failed: {error}"),
        }
    })?;
    let eigenvalues = device.upload(&inner.eigenvalues)?;
    let eigenvectors = device.upload(leto::Storage::as_slice(inner.eigenvectors.storage()))?;
    Ok(GpuSymmetricEigenDecomposition {
        inner,
        eigenvalues,
        eigenvectors,
        n,
    })
}

/// Compute only symmetric eigenvalues through the shared Leto provider.
pub fn symmetric_eigenvalues_jacobi(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<RocmBuffer<f32>> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let eigenvalues = leto_ops::symmetric_eigenvalues_jacobi(&view).map_err(|error| {
        HephaestusError::DispatchFailed {
            message: format!("symmetric eigenvalues failed: {error}"),
        }
    })?;
    device.upload(&eigenvalues)
}

/// Compute general eigenvalues, including complex conjugate pairs.
pub fn eigenvalues(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<RocmBuffer<Complex<f32>>> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        return device.alloc_zeroed::<Complex<f32>>(0);
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let eigenvalues =
        leto_ops::eigenvalues(&view).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("general eigenvalues failed: {error}"),
        })?;
    device.upload(&eigenvalues)
}
