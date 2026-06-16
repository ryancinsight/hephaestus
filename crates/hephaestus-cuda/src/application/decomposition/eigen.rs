//! GPU-resident Eigendecomposition.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};
use num_complex::Complex;

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Symmetric eigendecomposition result: device-resident eigenvalues and eigenvectors.
pub struct GpuSymmetricEigenDecomposition {
    #[allow(dead_code)]
    inner: leto_ops::SymmetricEigenDecomposition<f32>,
    eigenvalues: CudaBuffer<f32>,
    eigenvectors: CudaBuffer<f32>,
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
    pub fn eigenvalues(&self) -> &CudaBuffer<f32> {
        &self.eigenvalues
    }

    /// Borrow the eigenvectors buffer on the device.
    #[must_use]
    #[inline]
    pub fn eigenvectors(&self) -> &CudaBuffer<f32> {
        &self.eigenvectors
    }
}

/// Compute the symmetric eigendecomposition on the GPU.
pub fn symmetric_eigen_jacobi(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSymmetricEigenDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Symmetric eigendecomposition requires square matrix, got shape [{rows}, {cols}]"
            ),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 {
        let eigenvalues = device.alloc_zeroed::<f32>(0)?;
        let eigenvectors = device.alloc_zeroed::<f32>(0)?;
        let empty_view =
            leto::ArrayView::<f32, 2>::new(leto::Layout::c_contiguous([0, 0]).unwrap(), &[]);
        let inner = leto_ops::symmetric_eigen_jacobi(&empty_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("Symmetric eigendecomposition empty failed: {e}"),
            }
        })?;
        return Ok(GpuSymmetricEigenDecomposition {
            inner,
            eigenvalues,
            eigenvectors,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner =
        leto_ops::symmetric_eigen_jacobi(&view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Symmetric eigendecomposition failed: {e}"),
        })?;

    let val_slice = &inner.eigenvalues;
    let vec_slice = leto::Storage::as_slice(inner.eigenvectors.storage());
    let eigenvalues = device.upload(val_slice)?;
    let eigenvectors = device.upload(vec_slice)?;

    Ok(GpuSymmetricEigenDecomposition {
        inner,
        eigenvalues,
        eigenvectors,
        n: rows,
    })
}

/// Compute only the eigenvalues of a symmetric matrix on the GPU.
pub fn symmetric_eigenvalues_jacobi(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<CudaBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Symmetric eigenvalues require square matrix, got shape [{rows}, {cols}]"
            ),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let s_host = leto_ops::symmetric_eigenvalues_jacobi(&view).map_err(|e| {
        HephaestusError::DispatchFailed {
            message: format!("Symmetric eigenvalues failed: {e}"),
        }
    })?;

    device.upload(&s_host)
}

/// Compute the eigenvalues of a general (non-symmetric) matrix on the GPU, returning a complex buffer.
pub fn eigenvalues(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<CudaBuffer<Complex<f32>>> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Eigenvalues require square matrix, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 {
        return device.alloc_zeroed::<Complex<f32>>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let e_host = leto_ops::eigenvalues(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("General eigenvalues failed: {e}"),
    })?;

    device.upload(&e_host)
}
