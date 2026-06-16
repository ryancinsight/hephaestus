//! GPU-resident UDU decomposition.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// UDU decomposition result: device-resident factors.
pub struct GpuUduDecomposition {
    #[allow(dead_code)]
    inner: Option<leto_ops::UduDecomposition<f32>>,
    u: WgpuBuffer<f32>,
    d: WgpuBuffer<f32>,
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
    pub fn u_buffer(&self) -> &WgpuBuffer<f32> {
        &self.u
    }

    /// Borrow the diagonal factor **D** buffer on the device.
    #[must_use]
    #[inline]
    pub fn d_buffer(&self) -> &WgpuBuffer<f32> {
        &self.d
    }
}

/// Compute the UDU decomposition on the GPU.
pub fn udu_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuUduDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "UDU decomposition requires square matrix, got shape [{rows}, {cols}]"
            ),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    if rows == 0 {
        let u = device.alloc_zeroed::<f32>(0)?;
        let d = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuUduDecomposition {
            inner: None,
            u,
            d,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::udu_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("UDU decomposition failed: {e}"),
    })?;

    let u_host = inner.u();
    let u_slice = leto::Storage::as_slice(u_host.storage());
    let u = device.upload(u_slice)?;
    let d = device.upload(inner.diagonal())?;

    Ok(GpuUduDecomposition {
        inner: Some(inner),
        u,
        d,
        n: rows,
    })
}
