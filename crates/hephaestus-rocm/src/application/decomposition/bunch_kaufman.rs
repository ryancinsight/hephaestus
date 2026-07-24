//! ROCm Bunch–Kaufman decomposition backed by the shared Leto provider.
//!
//! The provider owns the symmetric-indefinite pivoting algorithm. ROCm keeps
//! the common device-resident L/D/permutation result contract and exposes no
//! CPU or WGPU backend-selection fallback.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// Bunch–Kaufman result with device-resident factors and host permutation.
pub struct GpuBunchKaufmanDecomposition {
    l: RocmBuffer<f32>,
    d: RocmBuffer<f32>,
    permutation: Vec<usize>,
    n: usize,
}

impl GpuBunchKaufmanDecomposition {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor **L** buffer on the device.
    #[must_use]
    #[inline]
    pub fn l_buffer(&self) -> &RocmBuffer<f32> {
        &self.l
    }

    /// Borrow the block-diagonal factor **D** buffer on the device.
    #[must_use]
    #[inline]
    pub fn d_buffer(&self) -> &RocmBuffer<f32> {
        &self.d
    }

    /// Return the symmetric pivot permutation.
    #[must_use]
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
}

/// Compute Bunch–Kaufman through the shared Leto provider and upload factors.
pub fn bunch_kaufman(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuBunchKaufmanDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        return Ok(GpuBunchKaufmanDecomposition {
            l: device.alloc_zeroed::<f32>(0)?,
            d: device.alloc_zeroed::<f32>(0)?,
            permutation: Vec::new(),
            n,
        });
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner =
        leto_ops::bunch_kaufman(&view).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("Bunch-Kaufman decomposition failed: {error}"),
        })?;
    let l = device.upload(leto::Storage::as_slice(inner.l().storage()))?;
    let d = device.upload(leto::Storage::as_slice(inner.d().storage()))?;
    let permutation = inner.permutation().to_vec();
    Ok(GpuBunchKaufmanDecomposition {
        l,
        d,
        permutation,
        n,
    })
}
