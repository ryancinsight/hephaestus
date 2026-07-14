//! GPU-resident Schur decomposition.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Schur decomposition result: device-resident factors.
pub struct GpuRealSchur {
    q: WgpuBuffer<f32>,
    t: WgpuBuffer<f32>,
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
    pub fn q_buffer(&self) -> &WgpuBuffer<f32> {
        &self.q
    }

    /// Borrow the real quasi-triangular **T** buffer on the device.
    #[must_use]
    #[inline]
    pub fn t_buffer(&self) -> &WgpuBuffer<f32> {
        &self.t
    }
}

/// Compute the real Schur decomposition on the GPU.
pub fn schur(device: &WgpuDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<GpuRealSchur> {
    let n = validate_square(&matrix)?;

    if n == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let t = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuRealSchur { q, t, n: 0 });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::schur(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Schur decomposition failed: {e}"),
    })?;

    let q_host = inner.q();
    let t_host = inner.t();
    let q_slice = leto::Storage::as_slice(q_host.storage());
    let t_slice = leto::Storage::as_slice(t_host.storage());
    let q = device.upload(q_slice)?;
    let t = device.upload(t_slice)?;

    Ok(GpuRealSchur { q, t, n })
}
