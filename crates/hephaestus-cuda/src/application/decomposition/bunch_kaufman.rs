//! GPU-resident Bunch-Kaufman decomposition.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Bunch-Kaufman decomposition result: device-resident factors.
pub struct GpuBunchKaufmanDecomposition {
    l: CudaBuffer<f32>,
    d: CudaBuffer<f32>,
    permutation: Vec<usize>,
    n: usize,
}

impl GpuBunchKaufmanDecomposition {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor **L** buffer on the device.
    #[must_use]
    #[inline]
    pub fn l_buffer(&self) -> &CudaBuffer<f32> {
        &self.l
    }

    /// Borrow the block-diagonal factor **D** buffer on the device.
    #[must_use]
    #[inline]
    pub fn d_buffer(&self) -> &CudaBuffer<f32> {
        &self.d
    }

    /// Return the permutation vector.
    #[must_use]
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
}

/// Compute the Bunch-Kaufman decomposition on the GPU.
pub fn bunch_kaufman(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuBunchKaufmanDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Bunch-Kaufman requires square matrix, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 {
        let l = device.alloc_zeroed::<f32>(0)?;
        let d = device.alloc_zeroed::<f32>(0)?;
        return Ok(GpuBunchKaufmanDecomposition {
            l,
            d,
            permutation: vec![],
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::bunch_kaufman(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Bunch-Kaufman decomposition failed: {e}"),
    })?;

    let l_host = inner.l();
    let d_host = inner.d();
    let l_slice = leto::Storage::as_slice(l_host.storage());
    let d_slice = leto::Storage::as_slice(d_host.storage());
    let l = device.upload(l_slice)?;
    let d = device.upload(d_slice)?;
    let permutation = inner.permutation().to_vec();

    Ok(GpuBunchKaufmanDecomposition {
        l,
        d,
        permutation,
        n: rows,
    })
}
