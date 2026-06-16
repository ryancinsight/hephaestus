//! GPU-resident symmetric eigendecomposition.
//!
//! Computes eigenpairs of a finite real symmetric matrix. The Jacobi
//! factorization is delegated to [`leto_ops`], and the resulting eigenvalues
//! and eigenvector matrix are stored on the device for downstream GPU
//! consumers.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use super::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Symmetric eigendecomposition result on the device.
///
/// Eigenvalues are sorted in ascending order. Eigenvectors are stored as
/// columns in the row-major `n x n` device buffer, matching Leto's
/// [`leto_ops::SymmetricEigenDecomposition`] contract.
pub struct GpuSymmetricEigenDecomposition {
    /// Host-side Leto decomposition that owns the canonical eigenpairs.
    inner: leto_ops::SymmetricEigenDecomposition<f32>,
    /// Device-resident eigenvalues sorted in ascending order.
    eigenvalues: WgpuBuffer<f32>,
    /// Device-resident row-major eigenvector matrix.
    eigenvectors: WgpuBuffer<f32>,
    n: usize,
}

impl GpuSymmetricEigenDecomposition {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the device-resident eigenvalue vector.
    #[must_use]
    #[inline]
    pub fn eigenvalues(&self) -> &WgpuBuffer<f32> {
        &self.eigenvalues
    }

    /// Borrow the device-resident eigenvector matrix.
    #[must_use]
    #[inline]
    pub fn eigenvectors(&self) -> &WgpuBuffer<f32> {
        &self.eigenvectors
    }

    /// Borrow the host-side Leto decomposition.
    ///
    /// This is retained so callers needing host-side scalar inspection can use
    /// the same canonical decomposition without downloading from the device.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::SymmetricEigenDecomposition<f32> {
        &self.inner
    }
}

/// Compute symmetric eigendecomposition for a finite square matrix.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite input.
/// - Input is not symmetric within Leto's Jacobi tolerance.
pub fn symmetric_eigen_jacobi(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSymmetricEigenDecomposition> {
    let n = validate_square(&matrix)?;
    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let eigen = leto_ops::symmetric_eigen_jacobi(&view).map_err(|e| {
        HephaestusError::DispatchFailed {
            message: format!("symmetric eigendecomposition failed: {e}"),
        }
    })?;

    let eigenvalues = device.upload(&eigen.eigenvalues)?;
    let eigenvectors = device.upload(leto::Storage::as_slice(eigen.eigenvectors.storage()))?;

    Ok(GpuSymmetricEigenDecomposition {
        inner: eigen,
        eigenvalues,
        eigenvectors,
        n,
    })
}

/// Compute only the symmetric eigenvalues for a finite square matrix.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite input.
/// - Input is not symmetric within Leto's Jacobi tolerance.
pub fn symmetric_eigenvalues_jacobi(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<WgpuBuffer<f32>> {
    let _n = validate_square(&matrix)?;
    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let eigenvalues = leto_ops::symmetric_eigenvalues_jacobi(&view).map_err(|e| {
        HephaestusError::DispatchFailed {
            message: format!("symmetric eigenvalue computation failed: {e}"),
        }
    })?;

    device.upload(&eigenvalues)
}
