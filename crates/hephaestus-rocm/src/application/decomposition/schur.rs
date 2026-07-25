//! ROCm real Schur decomposition backed by the shared Leto provider.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// Real Schur decomposition result with device-resident orthogonal and
/// quasi-triangular factors.
pub struct GpuRealSchur {
    q: RocmBuffer<f32>,
    t: RocmBuffer<f32>,
    n: usize,
}

impl GpuRealSchur {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the orthogonal Schur vectors **Q** buffer on the device.
    #[must_use]
    #[inline]
    pub fn q_buffer(&self) -> &RocmBuffer<f32> {
        &self.q
    }

    /// Borrow the real quasi-triangular factor **T** buffer on the device.
    #[must_use]
    #[inline]
    pub fn t_buffer(&self) -> &RocmBuffer<f32> {
        &self.t
    }
}

/// Compute the real Schur decomposition through the shared provider.
pub fn schur(device: &RocmDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<GpuRealSchur> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let t = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuRealSchur { q, t, n });
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::schur(&view).map_err(|error| HephaestusError::DispatchFailed {
        message: format!("Schur decomposition failed: {error}"),
    })?;

    let q_host = inner.q();
    let t_host = inner.t();
    let q_slice = leto::Storage::as_slice(q_host.storage());
    let t_slice = leto::Storage::as_slice(t_host.storage());
    let q = device.upload(q_slice)?;
    let t = device.upload(t_slice)?;
    Ok(GpuRealSchur { q, t, n })
}
