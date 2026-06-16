//! GPU-resident Golub-Kahan bidiagonalization.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Bidiagonal decomposition result: device-resident factors.
pub struct GpuBidiagonalDecomposition {
    #[allow(dead_code)]
    inner: leto_ops::BidiagonalDecomposition<f32>,
    u: WgpuBuffer<f32>,
    b: WgpuBuffer<f32>,
    v: WgpuBuffer<f32>,
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
    pub fn u_buffer(&self) -> &WgpuBuffer<f32> {
        &self.u
    }

    /// Borrow the bidiagonal factor **B** buffer on the device.
    #[must_use]
    #[inline]
    pub fn b_buffer(&self) -> &WgpuBuffer<f32> {
        &self.b
    }

    /// Borrow the orthogonal factor **V** buffer on the device.
    #[must_use]
    #[inline]
    pub fn v_buffer(&self) -> &WgpuBuffer<f32> {
        &self.v
    }
}

/// Compute the Golub-Kahan bidiagonalization on the GPU.
pub fn bidiagonalize(
    device: &WgpuDevice,
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
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    if m == 0 || n == 0 {
        let u = device.alloc_zeroed::<f32>(0)?;
        let b = device.alloc_zeroed::<f32>(0)?;
        let v = device.alloc_zeroed::<f32>(0)?;
        let placeholder = vec![0.0f32];
        let inner = leto_ops::bidiagonalize(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([1, 1]).unwrap(),
            &placeholder,
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Bidiagonalization failed: {e}"),
        })?;
        return Ok(GpuBidiagonalDecomposition {
            inner,
            u,
            b,
            v,
            m,
            n,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
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

    Ok(GpuBidiagonalDecomposition {
        inner,
        u,
        b,
        v,
        m,
        n,
    })
}
