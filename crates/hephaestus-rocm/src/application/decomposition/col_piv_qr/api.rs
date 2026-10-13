use hephaestus_core::{
    ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp, Result, map_layout_err,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_dense_operand;
use crate::application::pipeline::ColPivQrStage;
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;

use super::stages::checked_dimensions;
use super::stages::launch_stage;
use super::types::ColPivQrMeta;
use super::types::GpuColPivQrDecomposition;
use super::types::WIDTH;

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
