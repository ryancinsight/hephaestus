//! GPU-resident Schur decomposition.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Schur decomposition result: device-resident factors.
pub struct GpuRealSchur {
    #[allow(dead_code)]
    inner: Option<leto_ops::RealSchur<f32>>,
    q: CudaBuffer<f32>,
    t: CudaBuffer<f32>,
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
    pub fn q_buffer(&self) -> &CudaBuffer<f32> {
        &self.q
    }

    /// Borrow the real quasi-triangular **T** buffer on the device.
    #[must_use]
    #[inline]
    pub fn t_buffer(&self) -> &CudaBuffer<f32> {
        &self.t
    }
}

/// Compute the real Schur decomposition on the GPU.
pub fn schur(device: &CudaDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<GpuRealSchur> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Schur decomposition requires square matrix, got shape [{rows}, {cols}]"
            ),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let t = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuRealSchur {
            inner: None,
            q,
            t,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
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

    Ok(GpuRealSchur {
        inner: Some(inner),
        q,
        t,
        n: rows,
    })
}
