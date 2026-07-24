//! Numerical rank and determinant over ROCm device-resident matrices.

use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result};

use super::{map_layout_err, to_i32, to_u32};
use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// ROCm scalar supporting matrix-rank estimation and determinant calculation.
pub trait MatrixRankScalar: DialectScalar<HipC> + Pod {}

impl MatrixRankScalar for f32 {}

struct MatrixRankKernel<T>(PhantomData<T>);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RankMeta {
    shape: [u32; 2],
    strides: [i32; 2],
    offset: u32,
    tolerance: f32,
    _pad: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<RankMeta>() == 32);

fn shader_source<T: MatrixRankScalar>() -> String {
    format!(
        r#"
struct RankMeta {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
    float tolerance;
    unsigned int _pad[2];
}};

extern "C" __global__ void matrix_properties_kernel(
    RankMeta rank_meta,
    const {ty}* input,
    {ty}* scratch,
    unsigned int* rank_out,
    {ty}* det_out
) {{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index != 0u) {{
        return;
    }}

    unsigned int rows = rank_meta.shape[0];
    unsigned int cols = rank_meta.shape[1];
    if (rows == 0u || cols == 0u) {{
        rank_out[0] = 0u;
        det_out[0] = ({ty})0;
        return;
    }}

    bool square = rows == cols;
    {ty} max_abs = ({ty})0;
    unsigned int len = rows * cols;
    for (unsigned int linear = 0u; linear < len; linear++) {{
        unsigned int row = linear / cols;
        unsigned int col = linear - row * cols;
        int input_offset = (int)rank_meta.offset
            + (int)row * rank_meta.strides[0]
            + (int)col * rank_meta.strides[1];
        {ty} value = input[input_offset];
        scratch[linear] = value;
        {ty} absolute = value < ({ty})0 ? -value : value;
        if (absolute > max_abs) {{
            max_abs = absolute;
        }}
    }}

    if (max_abs <= ({ty})0) {{
        rank_out[0] = 0u;
        det_out[0] = ({ty})0;
        return;
    }}

    {ty} threshold = max_abs * ({ty})rank_meta.tolerance;
    unsigned int rank = 0u;
    {ty} determinant = ({ty})1;
    {ty} sign = ({ty})1;
    for (unsigned int col = 0u; col < cols; col++) {{
        if (rank >= rows) {{
            break;
        }}

        unsigned int pivot_row = rank;
        {ty} pivot_abs = ({ty})0;
        for (unsigned int row = rank; row < rows; row++) {{
            {ty} value = scratch[row * cols + col];
            {ty} absolute = value < ({ty})0 ? -value : value;
            if (absolute > pivot_abs) {{
                pivot_abs = absolute;
                pivot_row = row;
            }}
        }}

        if (pivot_abs > threshold) {{
            if (pivot_row != rank) {{
                sign = -sign;
                for (unsigned int swap_col = 0u; swap_col < cols; swap_col++) {{
                    unsigned int lhs = rank * cols + swap_col;
                    unsigned int rhs = pivot_row * cols + swap_col;
                    {ty} value = scratch[lhs];
                    scratch[lhs] = scratch[rhs];
                    scratch[rhs] = value;
                }}
            }}

            {ty} pivot = scratch[rank * cols + col];
            if (square) {{
                determinant = determinant * pivot;
            }}
            for (unsigned int row = 0u; row < rows; row++) {{
                if (row != rank) {{
                    {ty} factor = scratch[row * cols + col] / pivot;
                    for (unsigned int eliminate_col = col;
                         eliminate_col < cols; eliminate_col++) {{
                        unsigned int target = row * cols + eliminate_col;
                        unsigned int source = rank * cols + eliminate_col;
                        scratch[target] = scratch[target] - factor * scratch[source];
                    }}
                }}
            }}
            rank = rank + 1u;
        }}
    }}

    rank_out[0] = rank;
    if (square && rank == rows) {{
        det_out[0] = sign * determinant;
    }} else {{
        det_out[0] = ({ty})0;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

fn matrix_properties_with_tolerance<T: MatrixRankScalar>(
    device: &RocmDevice,
    matrix: StridedOperand<'_, T, 2>,
    relative_tolerance: f32,
) -> Result<(usize, RocmBuffer<T>)> {
    let [rows, cols] = matrix.layout.shape;
    if rows == 0 || cols == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matrix rank/det is undefined for empty matrix with shape {:?}",
                matrix.layout.shape
            ),
        });
    }
    if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matrix rank/det tolerance must be finite and non-negative, got {relative_tolerance}"
            ),
        });
    }

    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    let len = matrix.layout.checked_size().map_err(map_layout_err)?;
    let scratch = device.alloc_zeroed::<T>(len)?;
    let rank_out = device.alloc_zeroed::<u32>(1)?;
    let determinant = device.alloc_zeroed::<T>(1)?;
    let meta = RankMeta {
        shape: [
            to_u32(rows, "rank row count")?,
            to_u32(cols, "rank column count")?,
        ],
        strides: [
            to_i32(matrix.layout.strides[0], "rank row stride")?,
            to_i32(matrix.layout.strides[1], "rank column stride")?,
        ],
        offset: to_u32(matrix.layout.offset, "rank input offset")?,
        tolerance: relative_tolerance,
        _pad: [0; 2],
    };

    let key = PipelineKey::MatrixRank {
        marker: core::any::TypeId::of::<MatrixRankKernel<T>>(),
        scalar: core::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "matrix_properties_kernel", || {
        shader_source::<T>()
    })?;

    let mut meta = meta;
    let mut input_ptr: DevicePtr = matrix.buffer.raw();
    let mut scratch_ptr: DevicePtr = scratch.raw();
    let mut rank_ptr: DevicePtr = rank_out.raw();
    let mut determinant_ptr: DevicePtr = determinant.raw();
    let mut args: [*mut core::ffi::c_void; 5] = [
        (&mut meta as *mut RankMeta).cast(),
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut scratch_ptr as *mut DevicePtr).cast(),
        (&mut rank_ptr as *mut DevicePtr).cast(),
        (&mut determinant_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(device, &kernel, LaunchConfig::planar(1, 1, 1, 1), &mut args)?;

    let mut rank_value = [0_u32; 1];
    device.download(&rank_out, &mut rank_value)?;
    let rank = usize::try_from(rank_value[0]).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("matrix rank {} exceeds usize range", rank_value[0]),
    })?;
    Ok((rank, determinant))
}

/// Estimate the numerical rank of a finite rank-2 matrix on ROCm.
pub fn matrix_rank_with_tolerance<T: MatrixRankScalar>(
    device: &RocmDevice,
    matrix: StridedOperand<'_, T, 2>,
    relative_tolerance: f32,
) -> Result<usize> {
    matrix_properties_with_tolerance(device, matrix, relative_tolerance).map(|(rank, _)| rank)
}

/// Estimate the numerical rank of a finite rank-2 matrix on ROCm.
#[inline]
pub fn matrix_rank<T: MatrixRankScalar>(
    device: &RocmDevice,
    matrix: StridedOperand<'_, T, 2>,
) -> Result<usize> {
    matrix_rank_with_tolerance(device, matrix, 1.0e-9)
}

/// Compute the determinant of a finite square matrix on ROCm.
pub fn det<T: MatrixRankScalar>(
    device: &RocmDevice,
    matrix: StridedOperand<'_, T, 2>,
) -> Result<RocmBuffer<T>> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "det requires a square matrix, got shape {:?}",
                matrix.layout.shape
            ),
        });
    }
    matrix_properties_with_tolerance(device, matrix, 0.0).map(|(_, determinant)| determinant)
}

#[cfg(test)]
mod tests {
    use super::shader_source;

    #[test]
    fn source_declares_single_thread_row_reduction_contract() {
        let source = shader_source::<f32>();
        assert!(source.contains("matrix_properties_kernel"));
        assert!(source.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(source.contains("det_out[0] = sign * determinant"));
    }
}
