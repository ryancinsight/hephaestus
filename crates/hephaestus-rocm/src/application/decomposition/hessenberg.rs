//! ROCm Hessenberg reduction backed by the shared Leto provider.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// Hessenberg reduction result with device-resident orthogonal and reduced
/// factors.
pub struct GpuHessenbergDecomposition {
    q: RocmBuffer<f32>,
    h: RocmBuffer<f32>,
    n: usize,
}

impl GpuHessenbergDecomposition {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the orthogonal factor **Q** buffer on the device.
    #[must_use]
    #[inline]
    pub fn q_buffer(&self) -> &RocmBuffer<f32> {
        &self.q
    }

    /// Borrow the upper Hessenberg factor **H** buffer on the device.
    #[must_use]
    #[inline]
    pub fn h_buffer(&self) -> &RocmBuffer<f32> {
        &self.h
    }
}

/// Compute the upper Hessenberg reduction through the shared provider.
pub fn hessenberg(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuHessenbergDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let h = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuHessenbergDecomposition { q, h, n });
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::hessenberg(&view).map_err(|error| HephaestusError::DispatchFailed {
        message: format!("Hessenberg reduction failed: {error}"),
    })?;

    let q_host = inner.q();
    let h_host = inner.h();
    let q_slice = leto::Storage::as_slice(q_host.storage());
    let h_slice = leto::Storage::as_slice(h_host.storage());
    let q = device.upload(q_slice)?;
    let h = device.upload(h_slice)?;
    Ok(GpuHessenbergDecomposition { q, h, n })
}
