//! GPU-resident Golub-Kahan bidiagonalization.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Bidiagonal decomposition result: device-resident factors.
pub struct GpuBidiagonalDecomposition {
    u: CudaBuffer<f32>,
    b: CudaBuffer<f32>,
    v: CudaBuffer<f32>,
    m: usize,
    n: usize,
}

impl GpuBidiagonalDecomposition {
    /// Shape (rows, cols) of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Borrow the orthogonal factor **U** buffer on the device.
    #[must_use]
    #[inline]
    pub fn u_buffer(&self) -> &CudaBuffer<f32> {
        &self.u
    }

    /// Borrow the bidiagonal factor **B** buffer on the device.
    #[must_use]
    #[inline]
    pub fn b_buffer(&self) -> &CudaBuffer<f32> {
        &self.b
    }

    /// Borrow the orthogonal factor **V** buffer on the device.
    #[must_use]
    #[inline]
    pub fn v_buffer(&self) -> &CudaBuffer<f32> {
        &self.v
    }
}

/// Compute the Golub-Kahan bidiagonalization on the GPU.
pub fn bidiagonalize(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuBidiagonalDecomposition> {
    let [m, n] = matrix.layout.shape;
    if m < n {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Bidiagonalization requires m ≥ n, got shape [{m}, {n}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::bidiagonalize(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Bidiagonalization failed: {e}"),
    })?;

    let u_slice = leto::Storage::as_slice(inner.u().storage());
    let b_slice = leto::Storage::as_slice(inner.b().storage());
    let v_slice = leto::Storage::as_slice(inner.v().storage());
    let u = device.upload(u_slice)?;
    let b = device.upload(b_slice)?;
    let v = device.upload(v_slice)?;

    Ok(GpuBidiagonalDecomposition { u, b, v, m, n })
}
