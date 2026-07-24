//! ROCm bidiagonalization surface.
//!
//! The shared Leto operator owns the Golub–Kahan reduction used by the CUDA
//! and wgpu backends. ROCm keeps that provider boundary explicit, then uploads
//! the resulting orthogonal and bidiagonal factors into typed HIP buffers. No
//! backend-selection fallback is involved.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::RocmDevice;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("bidiagonalization layout error: {error}"),
    }
}

/// Bidiagonal factorization with device-resident **U**, **B**, and **V**.
pub struct GpuBidiagonalDecomposition {
    u: RocmBuffer<f32>,
    b: RocmBuffer<f32>,
    v: RocmBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuBidiagonalDecomposition {
    /// Shape `(rows, cols)` of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the left orthogonal factor **U** on the device.
    #[must_use]
    #[inline]
    pub fn u_buffer(&self) -> &RocmBuffer<f32> {
        &self.u
    }

    /// Borrow the upper-bidiagonal factor **B** on the device.
    #[must_use]
    #[inline]
    pub fn b_buffer(&self) -> &RocmBuffer<f32> {
        &self.b
    }

    /// Borrow the right orthogonal factor **V** on the device.
    #[must_use]
    #[inline]
    pub fn v_buffer(&self) -> &RocmBuffer<f32> {
        &self.v
    }
}

/// Compute Golub–Kahan bidiagonalization through the shared Leto provider.
pub fn bidiagonalize(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuBidiagonalDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Bidiagonalization requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner =
        leto_ops::bidiagonalize(&view).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("bidiagonalization failed: {error}"),
        })?;
    let u = device.upload(leto::Storage::as_slice(inner.u().storage()))?;
    let b = device.upload(leto::Storage::as_slice(inner.b().storage()))?;
    let v = device.upload(leto::Storage::as_slice(inner.v().storage()))?;
    Ok(GpuBidiagonalDecomposition {
        u,
        b,
        v,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn provider_boundary_is_explicit() {
        let source = include_str!("bidiagonal.rs");
        assert!(source.contains("leto_ops::bidiagonalize"));
        assert!(source.contains("typed HIP buffers"));
    }
}
