//! GPU-resident Hessenberg reduction.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Hessenberg reduction result: device-resident factors.
pub struct GpuHessenbergDecomposition {
    q: CudaBuffer<f32>,
    h: CudaBuffer<f32>,
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
    pub fn q_buffer(&self) -> &CudaBuffer<f32> {
        &self.q
    }

    /// Borrow the upper Hessenberg factor **H** buffer on the device.
    #[must_use]
    #[inline]
    pub fn h_buffer(&self) -> &CudaBuffer<f32> {
        &self.h
    }
}

/// Compute the upper Hessenberg reduction on the GPU.
pub fn hessenberg(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuHessenbergDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Hessenberg requires square matrix, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::hessenberg(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Hessenberg reduction failed: {e}"),
    })?;

    let q_slice = leto::Storage::as_slice(inner.q().storage());
    let h_slice = leto::Storage::as_slice(inner.h().storage());
    let q = device.upload(q_slice)?;
    let h = device.upload(h_slice)?;

    Ok(GpuHessenbergDecomposition { q, h, n: rows })
}
