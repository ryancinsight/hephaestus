//! Matrix multiplication on the CUDA device.
//!
//! A bespoke 16×16 shared-memory tiled kernel, authored as CUDA C generic over
//! the scalar (`T::TYPE_TOKEN` substitutes the device type token) and launched
//! directly through `cuLaunchKernel`. [`batched_matmul`] iterates the batch
//! dimension over [`matmul`], honoring batch broadcasting.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};
use leto::Layout;

use super::{map_layout, map_layout_err};
use crate::application::pipeline::{cached_kernel, launch_kernel, LaunchConfig};
use crate::application::strided::StridedOperand;
use crate::{CudaBuffer, CudaDevice};

struct MatmulKernel<T>(PhantomData<T>);

fn matmul_shader_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
}};

extern "C" __global__ void matmul_kernel(
    const {ty}* a,
    const {ty}* b,
    {ty}* c,
    MatrixLayout a_layout,
    MatrixLayout b_layout,
    MatrixLayout c_layout
) {{
    __shared__ {ty} A_shared[16][16];
    __shared__ {ty} B_shared[16][16];

    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;

    unsigned int m = a_layout.shape[0];
    unsigned int k = a_layout.shape[1];
    unsigned int n = b_layout.shape[1];

    int stride_a_row = a_layout.strides[0];
    int stride_a_col = a_layout.strides[1];
    int stride_b_row = b_layout.strides[0];
    int stride_b_col = b_layout.strides[1];

    {ty} sum = 0;
    unsigned int num_tiles = (k + 15u) / 16u;

    for (unsigned int tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {{
        // 1. Load A element into shared memory
        unsigned int col_a = tile_idx * 16u + local_col;
        if (row < m && col_a < k) {{
            int offset_a = (int)a_layout.offset + (int)row * stride_a_row + (int)col_a * stride_a_col;
            A_shared[local_row][local_col] = a[offset_a];
        }} else {{
            A_shared[local_row][local_col] = 0;
        }}

        // 2. Load B element into shared memory
        unsigned int row_b = tile_idx * 16u + local_row;
        if (row_b < k && col < n) {{
            int offset_b = (int)b_layout.offset + (int)row_b * stride_b_row + (int)col * stride_b_col;
            B_shared[local_row][local_col] = b[offset_b];
        }} else {{
            B_shared[local_row][local_col] = 0;
        }}

        __syncthreads();

        // 3. Accumulate product of the current tile
        for (unsigned int i = 0u; i < 16u; i++) {{
            sum += A_shared[local_row][i] * B_shared[i][local_col];
        }}

        __syncthreads();
    }}

    if (row < m && col < n) {{
        int stride_c_row = c_layout.strides[0];
        int stride_c_col = c_layout.strides[1];
        int offset_c = (int)c_layout.offset + (int)row * stride_c_row + (int)col * stride_c_col;
        c[offset_c] = sum;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

/// Perform matrix multiplication `out = lhs * rhs` on the CUDA device.
pub fn matmul_into<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [rows, lhs_shared] = lhs.layout.shape;
    let [rhs_shared, cols] = rhs.layout.shape;
    let [out_rows, out_cols] = out.layout.shape;

    if lhs_shared != rhs_shared || rows != out_rows || cols != out_cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape, rhs.layout.shape, out.layout.shape
            ),
        });
    }

    if lhs.buffer.aliases(out.buffer) || rhs.buffer.aliases(out.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }

    lhs.layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    out.layout
        .validate_storage_len(out.buffer.len())
        .map_err(map_layout_err)?;

    if out.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "matmul output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if rows == 0 || cols == 0 || lhs_shared == 0 {
        return Ok(());
    }

    let a_meta = map_layout(lhs.layout)?;
    let b_meta = map_layout(rhs.layout)?;
    let c_meta = map_layout(out.layout)?;

    let key = format!(
        "matmul_{}_{}",
        std::any::type_name::<MatmulKernel<T>>(),
        std::any::type_name::<T>()
    );

    let kernel = cached_kernel(device, key, "matmul_kernel", || matmul_shader_source::<T>())?;

    let workgroups_x = cols.div_ceil(16);
    let workgroups_y = rows.div_ceil(16);

    let mut a_ptr = lhs.buffer.raw();
    let mut b_ptr = rhs.buffer.raw();
    let mut c_ptr = out.buffer.raw();
    let mut a_meta_val = a_meta;
    let mut b_meta_val = b_meta;
    let mut c_meta_val = c_meta;

    // Argument list mirrors `matmul_kernel(const T*, const T*, T*, MatrixLayout,
    // MatrixLayout, MatrixLayout)`.
    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut c_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut a_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
        &mut b_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
        &mut c_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(workgroups_x as u32, workgroups_y as u32, 16, 16),
        &mut args,
    )
}

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` on the CUDA device.
pub fn batched_matmul_into<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
    out: StridedOperand<'_, T, 3>,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [lhs_batch, m, lhs_k] = lhs.layout.shape;
    let [rhs_batch, rhs_k, n] = rhs.layout.shape;
    let [out_batch, out_m, out_n] = out.layout.shape;

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k || m != out_m || n != out_n {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape, rhs.layout.shape, out.layout.shape
            ),
        });
    }

    lhs.layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    out.layout
        .validate_storage_len(out.buffer.len())
        .map_err(map_layout_err)?;

    let lhs_batch_stride = if lhs_batch == 1 {
        0
    } else {
        lhs.layout.strides[0]
    };
    let rhs_batch_stride = if rhs_batch == 1 {
        0
    } else {
        rhs.layout.strides[0]
    };
    let out_batch_stride = out.layout.strides[0];

    for b in 0..batch {
        let lhs_mat_layout = Layout::new(
            [m, lhs_k],
            [lhs.layout.strides[1], lhs.layout.strides[2]],
            (lhs.layout.offset as isize + b as isize * lhs_batch_stride) as usize,
        );
        let rhs_mat_layout = Layout::new(
            [rhs_k, n],
            [rhs.layout.strides[1], rhs.layout.strides[2]],
            (rhs.layout.offset as isize + b as isize * rhs_batch_stride) as usize,
        );
        let out_mat_layout = Layout::new(
            [out_m, out_n],
            [out.layout.strides[1], out.layout.strides[2]],
            (out.layout.offset as isize + b as isize * out_batch_stride) as usize,
        );

        let lhs_operand = StridedOperand {
            buffer: lhs.buffer,
            layout: &lhs_mat_layout,
        };
        let rhs_operand = StridedOperand {
            buffer: rhs.buffer,
            layout: &rhs_mat_layout,
        };
        let out_operand = StridedOperand {
            buffer: out.buffer,
            layout: &out_mat_layout,
        };

        matmul_into(device, lhs_operand, rhs_operand, out_operand)?;
    }

    Ok(())
}

/// Allocate and compute matrix multiplication `lhs * rhs` on the CUDA device.
///
/// The returned buffer has C-contiguous shape `[lhs.rows, rhs.cols]`.
pub fn matmul<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [rows, lhs_shared] = lhs.layout.shape;
    let [rhs_shared, cols] = rhs.layout.shape;
    if lhs_shared != rhs_shared {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape, rhs.layout.shape
            ),
        });
    }
    let out_layout = Layout::c_contiguous([rows, cols]).map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(out_layout.checked_size().map_err(map_layout_err)?)?;
    matmul_into(
        device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
    )?;
    Ok(out)
}

/// Allocate and compute batched matrix multiplication on the CUDA device.
///
/// Singleton batches broadcast to the other operand's batch count. The returned
/// buffer has C-contiguous shape `[batch, lhs.rows, rhs.cols]`.
pub fn batched_matmul<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [lhs_batch, m, lhs_k] = lhs.layout.shape;
    let [rhs_batch, rhs_k, n] = rhs.layout.shape;
    let batch = lhs_batch.max(rhs_batch);
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape, rhs.layout.shape
            ),
        });
    }
    let out_layout = Layout::c_contiguous([batch, m, n]).map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(out_layout.checked_size().map_err(map_layout_err)?)?;
    batched_matmul_into(
        device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
    )?;
    Ok(out)
}
