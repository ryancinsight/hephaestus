use hephaestus_core::{HephaestusError, Result};

use crate::RocmDevice;
use crate::application::pipeline::{
    ColPivQrStage, LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

use super::source::kernel_source;
use super::types::ColPivQrMeta;
use super::types::WIDTH;

pub(crate) fn checked_dimensions(
    rows: usize,
    cols: usize,
) -> Result<(u32, u32, u32, usize, usize)> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR element count overflows for [{rows}, {cols}]"),
        })?;
    let q_elements = rows
        .checked_mul(rows)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR Q size overflows for row count {rows}"),
        })?;
    let rows_u32 = u32::try_from(rows).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR row count {rows} exceeds HIP range"),
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR column count {cols} exceeds HIP range"),
    })?;
    let pivots_u32 =
        u32::try_from(rows.min(cols)).map_err(|_| HephaestusError::DispatchFailed {
            message: "column-pivoted QR pivot count exceeds HIP range".to_string(),
        })?;
    u32::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR element count {elements} exceeds HIP range"),
    })?;
    u32::try_from(q_elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR Q size {q_elements} exceeds HIP range"),
    })?;
    Ok((rows_u32, cols_u32, pivots_u32, elements, q_elements))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the parameters are the operation's independent descriptors; bundling them into a struct renames the same fields without removing a call-site decision"
)]
pub(crate) fn launch_stage(
    device: &RocmDevice,
    stage: ColPivQrStage,
    matrix: &RocmBuffer<f32>,
    q: &RocmBuffer<f32>,
    permutation: &RocmBuffer<u32>,
    heads: &RocmBuffer<f32>,
    betas: &RocmBuffer<f32>,
    status: &RocmBuffer<u32>,
    rank: &RocmBuffer<u32>,
    threshold: &RocmBuffer<f32>,
    meta: ColPivQrMeta,
    work_items: usize,
) -> Result<()> {
    let entry = match stage {
        ColPivQrStage::Validate => "col_piv_qr_validate",
        ColPivQrStage::Step => "col_piv_qr_step",
    };
    let kernel = cached_kernel(device, PipelineKey::ColPivQr(stage), entry, kernel_source)?;
    let grid = if matches!(stage, ColPivQrStage::Step) {
        1
    } else {
        grid_size(work_items, WIDTH)?
    };
    let mut matrix_ptr: DevicePtr = matrix.raw();
    let mut q_ptr: DevicePtr = q.raw();
    let mut permutation_ptr: DevicePtr = permutation.raw();
    let mut heads_ptr: DevicePtr = heads.raw();
    let mut betas_ptr: DevicePtr = betas.raw();
    let mut status_ptr: DevicePtr = status.raw();
    let mut rank_ptr: DevicePtr = rank.raw();
    let mut threshold_ptr: DevicePtr = threshold.raw();
    let mut meta = meta;
    let mut args: [*mut core::ffi::c_void; 9] = [
        (&mut matrix_ptr as *mut DevicePtr).cast(),
        (&mut q_ptr as *mut DevicePtr).cast(),
        (&mut permutation_ptr as *mut DevicePtr).cast(),
        (&mut heads_ptr as *mut DevicePtr).cast(),
        (&mut betas_ptr as *mut DevicePtr).cast(),
        (&mut status_ptr as *mut DevicePtr).cast(),
        (&mut rank_ptr as *mut DevicePtr).cast(),
        (&mut threshold_ptr as *mut DevicePtr).cast(),
        (&mut meta as *mut ColPivQrMeta).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, WIDTH),
        &mut args,
    )
}
