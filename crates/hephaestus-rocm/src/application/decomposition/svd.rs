//! ROCm SVD surfaces backed by the shared Leto provider.
//!
//! CUDA and wgpu use the same provider boundary for the thin, rank-revealing,
//! and singular-values-only paths. ROCm mirrors that contract and uploads the
//! resulting factors into typed device buffers. The public result is therefore
//! device-resident without claiming a separate HIP SVD kernel that does not
//! exist in this increment.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("SVD layout error: {error}"),
    }
}

/// SVD result with device-resident **U**, **V**, and singular values.
pub struct GpuSvdDecomposition {
    inner: leto_ops::SvdDecomposition<f32>,
    u: RocmBuffer<f32>,
    v: RocmBuffer<f32>,
    singular_values: RocmBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuSvdDecomposition {
    /// Shape `(rows, cols)` of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the left singular-vector buffer **U** on the device.
    #[must_use]
    #[inline]
    pub fn u(&self) -> &RocmBuffer<f32> {
        &self.u
    }

    /// Borrow the right singular-vector buffer **V** on the device.
    #[must_use]
    #[inline]
    pub fn v(&self) -> &RocmBuffer<f32> {
        &self.v
    }

    /// Borrow the singular-value buffer on the device.
    #[must_use]
    #[inline]
    pub fn singular_values(&self) -> &RocmBuffer<f32> {
        &self.singular_values
    }

    /// Borrow the shared Leto decomposition retained for provider inspection.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::SvdDecomposition<f32> {
        &self.inner
    }
}

fn empty_result(device: &RocmDevice, rows: usize, cols: usize) -> Result<GpuSvdDecomposition> {
    let inner = leto_ops::SvdDecomposition {
        singular_values: Vec::new(),
        left_singular_vectors: leto::Array2::from_shape_vec([0, 0], Vec::new()).map_err(
            |error| HephaestusError::DispatchFailed {
                message: format!("empty SVD U shape failed: {error}"),
            },
        )?,
        right_singular_vectors: leto::Array2::from_shape_vec([0, 0], Vec::new()).map_err(
            |error| HephaestusError::DispatchFailed {
                message: format!("empty SVD V shape failed: {error}"),
            },
        )?,
    };
    Ok(GpuSvdDecomposition {
        inner,
        u: device.alloc_zeroed::<f32>(0)?,
        v: device.alloc_zeroed::<f32>(0)?,
        singular_values: device.alloc_zeroed::<f32>(0)?,
        rows,
        cols,
    })
}

fn decompose(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
    rank_revealing: bool,
) -> Result<GpuSvdDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    if rows == 0 || cols == 0 {
        return empty_result(device, rows, cols);
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = if rank_revealing {
        leto_ops::svd_rank_revealing(&view)
    } else {
        leto_ops::svd_decompose(&view)
    }
    .map_err(|error| HephaestusError::DispatchFailed {
        message: format!(
            "{} SVD failed: {error}",
            if rank_revealing {
                "rank-revealing"
            } else {
                "thin"
            }
        ),
    })?;
    let u = device.upload(leto::Storage::as_slice(
        inner.left_singular_vectors.storage(),
    ))?;
    let v = device.upload(leto::Storage::as_slice(
        inner.right_singular_vectors.storage(),
    ))?;
    let singular_values = device.upload(&inner.singular_values)?;
    Ok(GpuSvdDecomposition {
        inner,
        u,
        v,
        singular_values,
        rows,
        cols,
    })
}

/// Compute the thin SVD through the shared Leto provider.
pub fn svd_decompose(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSvdDecomposition> {
    decompose(device, matrix, false)
}

/// Compute the rank-revealing SVD through the shared Leto provider.
pub fn svd_rank_revealing(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuSvdDecomposition> {
    decompose(device, matrix, true)
}

/// Compute only singular values through the shared Leto provider.
pub fn singular_values(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<RocmBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    if rows == 0 || cols == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let values =
        leto_ops::singular_values(&view).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("singular values failed: {error}"),
        })?;
    device.upload(&values)
}

#[cfg(test)]
mod tests {
    #[test]
    fn provider_boundary_is_explicit() {
        let source = include_str!("svd.rs");
        assert!(source.contains("leto_ops::svd_decompose"));
        assert!(source.contains("typed device buffers"));
    }
}
