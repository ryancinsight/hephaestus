//! GPU-resident SVD decomposition.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// SVD decomposition result: device-resident factors.
pub struct GpuSvdDecomposition {
    #[allow(dead_code)]
    inner: leto_ops::SvdDecomposition<f32>,
    u: WgpuBuffer<f32>,
    v: WgpuBuffer<f32>,
    singular_values: WgpuBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuSvdDecomposition {
    /// Shape (rows, cols) of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the left singular vectors **U** buffer on the device.
    #[must_use]
    #[inline]
    pub fn u(&self) -> &WgpuBuffer<f32> {
        &self.u
    }

    /// Borrow the right singular vectors **V** buffer on the device.
    #[must_use]
    #[inline]
    pub fn v(&self) -> &WgpuBuffer<f32> {
        &self.v
    }

    /// Borrow the singular values buffer on the device.
    #[must_use]
    #[inline]
    pub fn singular_values(&self) -> &WgpuBuffer<f32> {
        &self.singular_values
    }

    /// Borrow the host-side Leto decomposition.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::SvdDecomposition<f32> {
        &self.inner
    }
}

/// Compute the thin SVD decomposition on the GPU.
///
/// # Errors
/// - Empty/invalid shape.
/// - Non-finite values in the input.
/// - Rank-deficient input (rejected on the thin path).
pub fn svd_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSvdDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;
    if rows == 0 || cols == 0 {
        let u = device.alloc_zeroed::<f32>(0)?;
        let v = device.alloc_zeroed::<f32>(0)?;
        let s = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::SvdDecomposition {
            singular_values: vec![],
            left_singular_vectors: leto::Array2::from_shape_vec([0, 0], vec![]).unwrap(),
            right_singular_vectors: leto::Array2::from_shape_vec([0, 0], vec![]).unwrap(),
        };
        return Ok(GpuSvdDecomposition {
            inner,
            u,
            v,
            singular_values: s,
            rows,
            cols,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::svd_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("SVD decomposition failed: {e}"),
    })?;

    let u_slice = leto::Storage::as_slice(inner.left_singular_vectors.storage());
    let v_slice = leto::Storage::as_slice(inner.right_singular_vectors.storage());
    let u = device.upload(u_slice)?;
    let v = device.upload(v_slice)?;
    let s = device.upload(&inner.singular_values)?;

    Ok(GpuSvdDecomposition {
        inner,
        u,
        v,
        singular_values: s,
        rows,
        cols,
    })
}

/// Compute the rank-revealing SVD decomposition on the GPU.
///
/// # Errors
/// - Empty/invalid shape.
/// - Non-finite values in the input.
pub fn svd_rank_revealing(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSvdDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;
    if rows == 0 || cols == 0 {
        let u = device.alloc_zeroed::<f32>(0)?;
        let v = device.alloc_zeroed::<f32>(0)?;
        let s = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::SvdDecomposition {
            singular_values: vec![],
            left_singular_vectors: leto::Array2::from_shape_vec([0, 0], vec![]).unwrap(),
            right_singular_vectors: leto::Array2::from_shape_vec([0, 0], vec![]).unwrap(),
        };
        return Ok(GpuSvdDecomposition {
            inner,
            u,
            v,
            singular_values: s,
            rows,
            cols,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner =
        leto_ops::svd_rank_revealing(&view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("SVD rank-revealing failed: {e}"),
        })?;

    let u_slice = leto::Storage::as_slice(inner.left_singular_vectors.storage());
    let v_slice = leto::Storage::as_slice(inner.right_singular_vectors.storage());
    let u = device.upload(u_slice)?;
    let v = device.upload(v_slice)?;
    let s = device.upload(&inner.singular_values)?;

    Ok(GpuSvdDecomposition {
        inner,
        u,
        v,
        singular_values: s,
        rows,
        cols,
    })
}

/// Compute only the singular values on the GPU.
///
/// # Errors
/// - Empty/invalid shape.
/// - Non-finite values in the input.
pub fn singular_values(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<WgpuBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;
    if rows == 0 || cols == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let s_host = leto_ops::singular_values(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Singular values failed: {e}"),
    })?;

    device.upload(&s_host)
}
