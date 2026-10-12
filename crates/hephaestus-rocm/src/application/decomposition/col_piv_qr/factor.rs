use super::WIDTH;
use super::kernel::ColPivQrMeta;
use super::launch::{checked_dimensions, launch_stage};
use hephaestus_core::{
    ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp, Result, map_layout_err,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_dense_operand;
use crate::application::pipeline::ColPivQrStage;
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::RocmBuffer;

/// Column-pivoted QR result with device-backed **Q**, **R**, and permutation.
pub struct GpuColPivQrDecomposition {
    /// Host-side reference retained for scalar least-squares solves.
    inner: leto_ops::ColPivQrDecomposition<f32>,
    /// Materialized orthogonal factor **Q** on the device.
    q: RocmBuffer<f32>,
    /// Upper-triangular factor **R** on the device.
    r: RocmBuffer<f32>,
    /// Column permutation, where each entry is the original column at that position.
    permutation: Vec<usize>,
    rank: usize,
    rows: usize,
    cols: usize,
}

impl GpuColPivQrDecomposition {
    /// Numerical rank reported by the pivoted diagonal threshold.
    #[must_use]
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `(rows, cols)` of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the orthogonal factor **Q** buffer on the device.
    #[must_use]
    #[inline]
    pub fn q(&self) -> &RocmBuffer<f32> {
        &self.q
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r(&self) -> &RocmBuffer<f32> {
        &self.r
    }

    /// Return the column permutation.
    #[must_use]
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ using the retained factor.
    pub fn solve_least_squares(
        &self,
        device: &RocmDevice,
        rhs: &RocmBuffer<f32>,
    ) -> Result<RocmBuffer<f32>> {
        if rhs.len() != self.rows {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.rows,
                device_len: rhs.len(),
            });
        }
        if self.rows == 0 || self.cols == 0 {
            return device.upload(&[] as &[f32]);
        }

        let rhs_host = device.download_owned(rhs)?;
        let rhs_view = leto::ArrayView::<f32, 1>::new(
            Layout::c_contiguous([self.rows]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution = self.inner.solve_least_squares(&rhs_view).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR least-squares solve failed: {error}"),
            }
        })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }
}
/// Compute column-pivoted QR through HIP for a strided rank-2 matrix.
pub fn col_piv_qr(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuColPivQrDecomposition> {
    let [rows, cols] = matrix.layout.shape();
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    let (rows_u32, cols_u32, pivots_u32, elements, q_elements) = checked_dimensions(rows, cols)?;

    if rows == 0 || cols == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let r = device.alloc_zeroed::<f32>(0)?;
        let host_data = device.download_owned(matrix.buffer)?;
        let inner =
            leto_ops::col_piv_qr(&leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data))
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("empty column-pivoted QR decomposition failed: {error}"),
                })?;
        return Ok(GpuColPivQrDecomposition {
            inner,
            q,
            r,
            permutation: Vec::new(),
            rank: 0,
            rows,
            cols,
        });
    }

    let dense_layout =
        Layout::c_contiguous([rows, cols]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR dense layout failed: {error}"),
        })?;
    let r = device.alloc_uninitialized::<f32>(elements)?;
    unary_elementwise_strided_into::<IdentityOp, f32, 2>(
        device,
        matrix,
        StridedOperand {
            buffer: &r,
            layout: &dense_layout,
        },
        WIDTH,
    )?;
    let q = device.alloc_uninitialized::<f32>(q_elements)?;
    let permutation = device.alloc_uninitialized::<u32>(cols)?;
    let heads = device.alloc_zeroed::<f32>(rows.min(cols))?;
    let betas = device.alloc_zeroed::<f32>(rows.min(cols))?;
    let status = device.alloc_zeroed::<u32>(1)?;
    let rank = device.alloc_uninitialized::<u32>(1)?;
    let threshold = device.alloc_zeroed::<f32>(1)?;
    launch_stage(
        device,
        ColPivQrStage::Validate,
        &r,
        &q,
        &permutation,
        &heads,
        &betas,
        &status,
        &rank,
        &threshold,
        ColPivQrMeta {
            rows: rows_u32,
            cols: cols_u32,
            pivots: pivots_u32,
            k: 0,
        },
        elements.max(q_elements).max(cols),
    )?;
    for k in 0..rows.min(cols) {
        let k_u32 = u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR step {k} exceeds HIP range"),
        })?;
        launch_stage(
            device,
            ColPivQrStage::Step,
            &r,
            &q,
            &permutation,
            &heads,
            &betas,
            &status,
            &rank,
            &threshold,
            ColPivQrMeta {
                rows: rows_u32,
                cols: cols_u32,
                pivots: pivots_u32,
                k: k_u32,
            },
            1,
        )?;
    }

    let mut status_host = [0_u32; 1];
    device.download(&status, &mut status_host)?;
    if status_host[0] != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: "column-pivoted QR factorization failed: input is non-finite or unstable"
                .to_string(),
        });
    }
    let mut rank_host = [0_u32; 1];
    device.download(&rank, &mut rank_host)?;
    let rank = usize::try_from(rank_host[0]).map_err(|_| HephaestusError::DispatchFailed {
        message: "column-pivoted QR rank exceeds host index range".to_string(),
    })?;
    let permutation_host = device.download_owned(&permutation)?;
    let permutation = permutation_host
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
                message: "column-pivoted QR permutation exceeds host range".to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let host_data = device.download_owned(matrix.buffer)?;
    let inner = leto_ops::col_piv_qr(&leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data))
        .map_err(|error| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR host contract failed: {error}"),
    })?;
    Ok(GpuColPivQrDecomposition {
        inner,
        q,
        r,
        permutation,
        rank,
        rows,
        cols,
    })
}

/// Validate a dense C-contiguous matrix before computing column-pivoted QR.
pub fn col_piv_qr_blocked(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuColPivQrDecomposition> {
    validate_dense_operand("column-pivoted QR", &matrix)?;
    col_piv_qr(device, matrix)
}
