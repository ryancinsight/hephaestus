//! GPU-resident LU decomposition with partial pivoting.
//!
//! Computes **P A** = **L U** where **P** is a permutation matrix, **L** is
//! unit lower-triangular, and **U** is upper-triangular. Delegates to the
//! CPU via [`leto_ops`], storing the result on the device and as a host-side
//! decomposition for solve/inv/det without re-factorization.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use super::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// LU decomposition result: device-resident packed factors with host-side
/// decomposition for solve/inv/det.
pub struct GpuLuDecomposition {
    /// Host-side leto-ops decomposition (owns pivots, sign, factors).
    inner: leto_ops::LuDecomposition<f32>,
    /// Device-resident packed L/U factors (*n* × *n*, row-major).
    factors: CudaBuffer<f32>,
    n: usize,
}

impl GpuLuDecomposition {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the packed factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn factors(&self) -> &CudaBuffer<f32> {
        &self.factors
    }

    /// Determinant: sign × Πᵢ Uᵢᵢ via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side forward/back substitution.
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
                message: format!("LU solve failed: {e}"),
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
                message: format!("LU inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

/// Compute the LU decomposition with partial pivoting on the GPU.
///
/// # Errors
///
/// - Non-square input.
/// - Non-finite values in the input.
/// - Singular matrix (exact zero pivot).
pub fn lu_decompose(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let factors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::lu_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([0, 0]).unwrap(),
            &[],
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("LU decomposition failed: {e}"),
        })?;
        return Ok(GpuLuDecomposition {
            inner,
            factors,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let lu = leto_ops::lu_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("LU decomposition failed: {e}"),
    })?;

    let factors = device.upload(leto::Storage::as_slice(lu.factors().storage()))?;

    Ok(GpuLuDecomposition {
        inner: lu,
        factors,
        n,
    })
}
