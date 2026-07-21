//! GPU-resident UDU decomposition.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// UDU decomposition result: device-resident factors.
pub struct GpuUduDecomposition {
    inner: Option<leto_ops::UduDecomposition<f32>>,
    u: CudaBuffer<f32>,
    d: CudaBuffer<f32>,
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
    pub fn u_buffer(&self) -> &CudaBuffer<f32> {
        &self.u
    }

    /// Borrow the diagonal factor **D** buffer on the device.
    #[must_use]
    #[inline]
    pub fn d_buffer(&self) -> &CudaBuffer<f32> {
        &self.d
    }

    /// Determinant `det(A) = product(D[i])` via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner
            .as_ref()
            .map_or(1.0, leto_ops::UduDecomposition::det)
    }

    /// Solve **A** · **x** = **rhs** via the stored host-side decomposition.
    pub fn solve(&self, device: &CudaDevice, rhs: &CudaBuffer<f32>) -> Result<CudaBuffer<f32>> {
        if rhs.len() != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len(),
            });
        }
        if self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let inner = self
            .inner
            .as_ref()
            .expect("invariant: non-empty UDU decomposition stores host factors");
        let mut rhs_host = vec![0.0f32; self.n];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([self.n]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
        let x = inner
            .solve(&rhs_view)
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("UDU solve failed: {e}"),
            })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }

    /// Compute the inverse **A**⁻¹ via the stored host-side decomposition.
    pub fn inv(&self, device: &CudaDevice) -> Result<CudaBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inner = self
            .inner
            .as_ref()
            .expect("invariant: non-empty UDU decomposition stores host factors");
        let inv = inner.inv().map_err(|e| HephaestusError::DispatchFailed {
            message: format!("UDU inverse failed: {e}"),
        })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

/// Compute the UDU decomposition on the GPU.
pub fn udu_decompose(
    device: &CudaDevice,
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
        .validate_storage_len(matrix.buffer.len())
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

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
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
