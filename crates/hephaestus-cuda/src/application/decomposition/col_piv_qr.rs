//! GPU-resident Column-Pivoted QR decomposition.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Column-pivoted QR decomposition result: device-resident factors.
pub struct GpuColPivQrDecomposition {
    inner: leto_ops::ColPivQrDecomposition<f32>,
    q: CudaBuffer<f32>,
    r: CudaBuffer<f32>,
    permutation: Vec<usize>,
    rank: usize,
    m: usize,
    n: usize,
}

impl GpuColPivQrDecomposition {
    /// Numerical rank (count of above-threshold R diagonal entries).
    #[must_use]
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Borrow the orthogonal factor **Q** buffer on the device.
    #[must_use]
    #[inline]
    pub fn q(&self) -> &CudaBuffer<f32> {
        &self.q
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r(&self) -> &CudaBuffer<f32> {
        &self.r
    }

    /// Return the column permutation.
    #[must_use]
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ (least squares).
    pub fn solve_least_squares(
        &self,
        device: &CudaDevice,
        rhs: &CudaBuffer<f32>,
    ) -> Result<CudaBuffer<f32>> {
        if rhs.len() != self.m {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.m,
                device_len: rhs.len(),
            });
        }
        if self.m == 0 || self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0f32; self.m];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([self.m]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
        let x = self.inner.solve_least_squares(&rhs_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("ColPivQR least-squares solve failed: {e}"),
            }
        })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }
}

/// Compute the column-pivoted QR decomposition on the GPU.
pub fn col_piv_qr(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuColPivQrDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::col_piv_qr(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("ColPivQR decomposition failed: {e}"),
    })?;

    let q_host = inner.q();
    let r_host = inner.r();
    let q = device.upload(leto::Storage::as_slice(q_host.storage()))?;
    let r = device.upload(leto::Storage::as_slice(r_host.storage()))?;
    let permutation = inner.permutation().to_vec();
    let rank = inner.rank();

    Ok(GpuColPivQrDecomposition {
        inner,
        q,
        r,
        permutation,
        rank,
        m: rows,
        n: cols,
    })
}
