//! GPU-resident complete pivoted LU decomposition.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Complete-pivoted LU decomposition result: device-resident factors.
pub struct GpuFullPivLuDecomposition {
    inner: leto_ops::FullPivLuDecomposition<f32>,
    lu: CudaBuffer<f32>,
    row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    rank: usize,
    n: usize,
}

impl GpuFullPivLuDecomposition {
    /// Dimension of the square matrix.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Numerical rank (count of nonzero pivots).
    #[must_use]
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Borrow the packed L/U factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lu_buffer(&self) -> &CudaBuffer<f32> {
        &self.lu
    }

    /// Return the row permutation vector.
    #[must_use]
    #[inline]
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_perm
    }

    /// Return the column permutation vector.
    #[must_use]
    #[inline]
    pub fn col_permutation(&self) -> &[usize] {
        &self.col_perm
    }

    /// Compute the determinant via host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side substitution.
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

        let mut rhs_host = vec![0.0f32; self.n];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([self.n]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
        let x = self
            .inner
            .solve(&rhs_view)
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("FullPivLU solve failed: {e}"),
            })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }

    /// Compute the inverse **A**⁻¹ via host-side decomposition.
    pub fn inv(&self, device: &CudaDevice) -> Result<CudaBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inv = self
            .inner
            .inv()
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("FullPivLU inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

/// Compute the complete-pivoted LU decomposition on the GPU.
pub fn full_piv_lu(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuFullPivLuDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("FullPivLU requires square matrix, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let inner = leto_ops::full_piv_lu(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("FullPivLU decomposition failed: {e}"),
    })?;

    let lu = device.upload(inner.lu_factors())?;
    let row_perm = inner.row_permutation().to_vec();
    let col_perm = inner.col_permutation().to_vec();
    let rank = inner.rank();

    Ok(GpuFullPivLuDecomposition {
        inner,
        lu,
        row_perm,
        col_perm,
        rank,
        n: rows,
    })
}
