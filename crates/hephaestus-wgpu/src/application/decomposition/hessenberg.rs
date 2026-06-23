//! GPU-resident Hessenberg reduction.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::decomposition::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Hessenberg reduction result: device-resident factors.
pub struct GpuHessenbergDecomposition {
    #[allow(dead_code)]
    inner: Option<leto_ops::HessenbergDecomposition<f32>>,
    q: WgpuBuffer<f32>,
    h: WgpuBuffer<f32>,
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
    pub fn q_buffer(&self) -> &WgpuBuffer<f32> {
        &self.q
    }

    /// Borrow the upper Hessenberg factor **H** buffer on the device.
    #[must_use]
    #[inline]
    pub fn h_buffer(&self) -> &WgpuBuffer<f32> {
        &self.h
    }
}

/// Compute the upper Hessenberg reduction on the GPU.
pub fn hessenberg(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuHessenbergDecomposition> {
    let n = validate_square(&matrix)?;

    if n == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let h = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuHessenbergDecomposition { inner: None, q, h, n: 0 });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::hessenberg(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Hessenberg reduction failed: {e}"),
    })?;

    let q_slice = leto::Storage::as_slice(inner.q().storage());
    let h_slice = leto::Storage::as_slice(inner.h().storage());
    let q = device.upload(q_slice)?;
    let h = device.upload(h_slice)?;

    Ok(GpuHessenbergDecomposition {
        inner: Some(inner),
        q,
        h,
        n,
    })
}
