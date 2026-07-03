//! Numerical rank and determinant of a matrix on the CUDA device.

use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};

use super::{map_layout_err, to_i32, to_u32};
use crate::application::pipeline::{cached_kernel, launch_kernel, LaunchConfig};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

/// CUDA scalar supporting matrix-rank and determinant estimation.
pub trait MatrixRankScalar: DialectScalar<CudaC> + Pod {}

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

fn matrix_properties_shader_source<T: MatrixRankScalar>() -> String {
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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0u) {{
        return;
    }}

    unsigned int rows = rank_meta.shape[0];
    unsigned int cols = rank_meta.shape[1];
    if (rows == 0u || cols == 0u) {{
        rank_out[0] = 0u;
        det_out[0] = ({ty})0.0;
        return;
    }}

    bool square = rows == cols;
    {ty} max_abs = 0.0;
    unsigned int len = rows * cols;
    for (unsigned int idx = 0u; idx < len; idx++) {{
        unsigned int row = idx / cols;
        unsigned int col = idx - row * cols;
        int input_offset = (int)rank_meta.offset
            + (int)row * rank_meta.strides[0]
            + (int)col * rank_meta.strides[1];
        {ty} value = input[input_offset];
        scratch[idx] = value;
        {ty} a_val = value;
        if (a_val < 0.0) {{
            a_val = -a_val;
        }}
        if (a_val > max_abs) {{
            max_abs = a_val;
        }}
    }}

    if (max_abs <= 0.0) {{
        rank_out[0] = 0u;
        det_out[0] = ({ty})0.0;
        return;
    }}

    {ty} threshold = max_abs * rank_meta.tolerance;
    unsigned int rank = 0u;
    {ty} det = ({ty})1.0;
    {ty} sign = ({ty})1.0;
    for (unsigned int col = 0u; col < cols; col++) {{
        if (rank >= rows) {{
            break;
        }}

        unsigned int pivot_row = rank;
        {ty} pivot_abs = 0.0;
        for (unsigned int row = rank; row < rows; row++) {{
            {ty} val = scratch[row * cols + col];
            if (val < 0.0) {{
                val = -val;
            }}
            if (val > pivot_abs) {{
                pivot_abs = val;
                pivot_row = row;
            }}
        }}

        if (pivot_abs > threshold) {{
            if (pivot_row != rank) {{
                sign = -sign;
                for (unsigned int swap_col = 0u; swap_col < cols; swap_col++) {{
                    unsigned int lhs = rank * cols + swap_col;
                    unsigned int rhs = pivot_row * cols + swap_col;
                    {ty} tmp = scratch[lhs];
                    scratch[lhs] = scratch[rhs];
                    scratch[rhs] = tmp;
                }}
            }}

            {ty} pivot = scratch[rank * cols + col];
            if (square) {{
                det = det * pivot;
            }}
            for (unsigned int row = 0u; row < rows; row++) {{
                if (row != rank) {{
                    {ty} factor = scratch[row * cols + col] / pivot;
                    for (unsigned int elim_col = col; elim_col < cols; elim_col++) {{
                        unsigned int target_idx = row * cols + elim_col;
                        unsigned int source = rank * cols + elim_col;
                        scratch[target_idx] = scratch[target_idx] - factor * scratch[source];
                    }}
                }}
            }}
            rank = rank + 1u;
        }}
    }}

    rank_out[0] = rank;
    if (square && rank == rows) {{
        det_out[0] = sign * det;
    }} else {{
        det_out[0] = ({ty})0.0;
    }}
}}
"#,
        ty = T::TYPE_TOKEN
    )
}

fn matrix_properties_with_tolerance<T>(
    device: &CudaDevice,
    matrix: StridedOperand<'_, T, 2>,
    relative_tolerance: f32,
) -> Result<(usize, CudaBuffer<T>)>
where
    T: MatrixRankScalar,
{
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
    let det_out = device.alloc_zeroed::<T>(1)?;
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

    let key = format!(
        "matrix_properties_{}_{}",
        std::any::type_name::<MatrixRankKernel<T>>(),
        std::any::type_name::<T>()
    );

    let kernel = cached_kernel(device, key, "matrix_properties_kernel", || {
        matrix_properties_source::<T>()
    })?;

    let mut meta_val = meta;
    let mut in_ptr = matrix.buffer.raw();
    let mut scratch_ptr = scratch.raw();
    let mut rank_ptr = rank_out.raw();
    let mut det_ptr = det_out.raw();

    // Argument list mirrors `matrix_properties_kernel(RankMeta, const T*, T*,
    // unsigned int*, T*)`.
    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut meta_val as *mut RankMeta as *mut std::ffi::c_void,
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut scratch_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut rank_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut det_ptr as *mut u64 as *mut std::ffi::c_void,
    ];

    // Sequential Gaussian elimination: a single-thread launch by design.
    launch_kernel(device, &kernel, LaunchConfig::planar(1, 1, 1, 1), &mut args)?;

    let mut rank = [0u32; 1];
    device.download(&rank_out, &mut rank)?;
    let rank = usize::try_from(rank[0]).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("matrix rank {} exceeds usize range", rank[0]),
    })?;

    Ok((rank, det_out))
}

fn matrix_properties_source<T: MatrixRankScalar>() -> String {
    matrix_properties_shader_source::<T>()
}

/// Estimate the numerical rank of a finite rank-2 matrix on the CUDA device.
pub fn matrix_rank_with_tolerance<T>(
    device: &CudaDevice,
    matrix: StridedOperand<'_, T, 2>,
    relative_tolerance: f32,
) -> Result<usize>
where
    T: MatrixRankScalar,
{
    matrix_properties_with_tolerance(device, matrix, relative_tolerance).map(|(rank, _)| rank)
}

/// Estimate the numerical rank of a finite rank-2 matrix on the CUDA device.
#[inline]
pub fn matrix_rank<T>(device: &CudaDevice, matrix: StridedOperand<'_, T, 2>) -> Result<usize>
where
    T: MatrixRankScalar,
{
    matrix_rank_with_tolerance(device, matrix, 1.0e-9)
}

/// Compute the determinant of a finite square matrix on the CUDA device.
pub fn det<T>(device: &CudaDevice, matrix: StridedOperand<'_, T, 2>) -> Result<CudaBuffer<T>>
where
    T: MatrixRankScalar,
{
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
