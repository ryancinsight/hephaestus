//! GPU-resident Eigendecomposition.

use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Complex;

/// Symmetric eigendecomposition result: device-resident eigenvalues and eigenvectors.
pub struct GpuSymmetricEigenDecomposition {
    inner: leto_ops::SymmetricEigenDecomposition<f32>,
    eigenvalues: WgpuBuffer<f32>,
    eigenvectors: WgpuBuffer<f32>,
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
    pub fn eigenvalues(&self) -> &WgpuBuffer<f32> {
        &self.eigenvalues
    }

    /// Borrow the eigenvectors buffer on the device.
    #[must_use]
    #[inline]
    pub fn eigenvectors(&self) -> &WgpuBuffer<f32> {
        &self.eigenvectors
    }

    /// Borrow the host-side Leto decomposition.
    ///
    /// This exposes the canonical host eigenpairs without requiring a device
    /// download when callers need scalar-side inspection.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::SymmetricEigenDecomposition<f32> {
        &self.inner
    }
}

/// Compute the symmetric eigendecomposition on the GPU.
pub fn symmetric_eigen_jacobi(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSymmetricEigenDecomposition> {
    let n = validate_square(&matrix)?;

    if n == 0 {
        let eigenvalues = device.alloc_zeroed::<f32>(0)?;
        let eigenvectors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::SymmetricEigenDecomposition {
            eigenvalues: vec![],
            eigenvectors: leto::Array2::from_shape_vec([0, 0], vec![])
                .expect("invariant: shape [0,0] is consistent with an empty vec"),
        };
        return Ok(GpuSymmetricEigenDecomposition {
            inner,
            eigenvalues,
            eigenvectors,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
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
        n,
    })
}

/// Compute only the eigenvalues of a symmetric matrix on the GPU.
pub fn symmetric_eigenvalues_jacobi(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<WgpuBuffer<f32>> {
    let n = validate_square(&matrix)?;

    if n == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
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
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<WgpuBuffer<Complex<f32>>> {
    let n = validate_square(&matrix)?;

    if n == 0 {
        return device.alloc_zeroed::<Complex<f32>>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let e_host = leto_ops::eigenvalues(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("General eigenvalues failed: {e}"),
    })?;

    device.upload(&e_host)
}
