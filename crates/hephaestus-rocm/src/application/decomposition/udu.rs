//! ROCm UDU decomposition backed by the shared Leto provider.
//!
//! UDU is a symmetric-indefinite factorization. ROCm mirrors the CUDA and
//! wgpu device-resident result contract by uploading the provider's upper
//! factor and diagonal; solve, determinant, and inverse retain the common
//! host-operation contract without a backend-selection fallback.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// UDU decomposition result with device-resident upper factor and diagonal.
pub struct GpuUduDecomposition {
    inner: Option<leto_ops::UduDecomposition<f32>>,
    u: RocmBuffer<f32>,
    d: RocmBuffer<f32>,
    n: usize,
}

impl GpuUduDecomposition {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the upper-triangular factor **U** buffer on the device.
    #[must_use]
    #[inline]
    pub fn u_buffer(&self) -> &RocmBuffer<f32> {
        &self.u
    }

    /// Borrow the diagonal factor **D** buffer on the device.
    #[must_use]
    #[inline]
    pub fn d_buffer(&self) -> &RocmBuffer<f32> {
        &self.d
    }

    /// Compute `det(A) = product(D[i])` from the retained provider factor.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner
            .as_ref()
            .map_or(1.0, leto_ops::UduDecomposition::det)
    }

    /// Solve **A** · **x** = **rhs** through the retained provider factor.
    pub fn solve(&self, device: &RocmDevice, rhs: &RocmBuffer<f32>) -> Result<RocmBuffer<f32>> {
        if rhs.len() != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len(),
            });
        }
        if self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: "UDU factor missing for non-empty matrix".to_string(),
            })?;
        let mut rhs_host = vec![0.0_f32; self.n];
        device.download(rhs, &mut rhs_host)?;
        let layout = leto::Layout::c_contiguous([self.n]).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("UDU RHS layout failed: {error}"),
            }
        })?;
        let rhs_view = leto::ArrayView::<f32, 1>::new(layout, &rhs_host);
        let solution = inner
            .solve(&rhs_view)
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("UDU solve failed: {error}"),
            })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }

    /// Compute **A**⁻¹ through the retained provider factor.
    pub fn inv(&self, device: &RocmDevice) -> Result<RocmBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: "UDU factor missing for non-empty matrix".to_string(),
            })?;
        let inverse = inner
            .inv()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("UDU inverse failed: {error}"),
            })?;
        device.upload(leto::Storage::as_slice(inverse.storage()))
    }
}

/// Compute UDU through the shared Leto provider and upload its factors.
pub fn udu_decompose(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuUduDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        return Ok(GpuUduDecomposition {
            inner: None,
            u: device.alloc_zeroed::<f32>(0)?,
            d: device.alloc_zeroed::<f32>(0)?,
            n,
        });
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner =
        leto_ops::udu_decompose(&view).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("UDU decomposition failed: {error}"),
        })?;
    let u = device.upload(leto::Storage::as_slice(inner.u().storage()))?;
    let d = device.upload(inner.diagonal())?;
    Ok(GpuUduDecomposition {
        inner: Some(inner),
        u,
        d,
        n,
    })
}
