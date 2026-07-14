//! Matrix multiplication on the CUDA device.
//!
//! A bespoke 16×16 shared-memory tiled kernel, authored as CUDA C generic over
//! the scalar (`T::TYPE_TOKEN` substitutes the device type token) and launched
//! directly through `cuLaunchKernel`. [`batched_matmul`] dispatches the batch
//! dimension via `blockIdx.z` in a dedicated kernel (CU-P5) rather than
//! looping [`matmul_into`] once per batch element — each loop iteration was a
//! separate `cuLaunchKernel` plus (on Windows) a `cuCtxSynchronize` context
//! drain per `launch_kernel` call (KS-8), serializing what should be one
//! dispatch. `MAX_GRID_Z` chunks batches beyond CUDA's hardware grid.z limit
//! (65535 on every current compute capability) into multiple launches.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};
use leto::Layout;

use super::{map_layout, map_layout_err, to_i64};
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::{CudaBuffer, CudaDevice};

/// CUDA's hardware grid.z dimension limit (fixed at 65535 across every
/// current compute capability, unlike grid.x/y which scale with it).
const MAX_GRID_Z: usize = 65_535;

struct MatmulKernel<T>(PhantomData<T>);
struct BatchedMatmulKernel<T>(PhantomData<T>);

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

    let key = PipelineKey::Matmul {
        marker: std::any::TypeId::of::<MatmulKernel<T>>(),
        scalar: std::any::TypeId::of::<T>(),
    };

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

fn batched_matmul_shader_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
}};

extern "C" __global__ void batched_matmul_kernel(
    const {ty}* a,
    const {ty}* b,
    {ty}* c,
    MatrixLayout a_layout,
    MatrixLayout b_layout,
    MatrixLayout c_layout,
    long long a_batch_stride,
    long long b_batch_stride,
    long long c_batch_stride,
    unsigned int batch_offset
) {{
    __shared__ {ty} A_shared[16][16];
    __shared__ {ty} B_shared[16][16];

    unsigned int batch_idx = batch_offset + blockIdx.z;
    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;

    unsigned int m = a_layout.shape[0];
    unsigned int k = a_layout.shape[1];
    unsigned int n = b_layout.shape[1];

    long long a_batch_off = (long long)batch_idx * a_batch_stride;
    long long b_batch_off = (long long)batch_idx * b_batch_stride;
    long long c_batch_off = (long long)batch_idx * c_batch_stride;

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
            long long offset_a = a_batch_off + (long long)a_layout.offset
                + (long long)row * stride_a_row + (long long)col_a * stride_a_col;
            A_shared[local_row][local_col] = a[offset_a];
        }} else {{
            A_shared[local_row][local_col] = 0;
        }}

        // 2. Load B element into shared memory
        unsigned int row_b = tile_idx * 16u + local_row;
        if (row_b < k && col < n) {{
            long long offset_b = b_batch_off + (long long)b_layout.offset
                + (long long)row_b * stride_b_row + (long long)col * stride_b_col;
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
        long long offset_c = c_batch_off + (long long)c_layout.offset
            + (long long)row * stride_c_row + (long long)col * stride_c_col;
        c[offset_c] = sum;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` on the CUDA device.
///
/// Dispatches the batch dimension via `blockIdx.z` in one kernel launch (or
/// the minimum number of launches CUDA's grid.z limit requires — batches
/// beyond `MAX_GRID_Z` chunk into further launches, each still covering up
/// to 65535 batch elements) rather than one `matmul_into` launch per batch
/// element. `lhs`/`rhs` broadcast when their batch dimension is 1.
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

    if batch == 0 || m == 0 || n == 0 || lhs_k == 0 {
        return Ok(());
    }

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

    // Per-matrix layout at batch index 0; the kernel adds `batch_idx *
    // {a,b,c}_batch_stride` on the device side, so the host only maps the
    // base (rank-2) slice once.
    let lhs_mat_layout = Layout::new(
        [m, lhs_k],
        [lhs.layout.strides[1], lhs.layout.strides[2]],
        lhs.layout.offset,
    );
    let rhs_mat_layout = Layout::new(
        [rhs_k, n],
        [rhs.layout.strides[1], rhs.layout.strides[2]],
        rhs.layout.offset,
    );
    let out_mat_layout = Layout::new(
        [out_m, out_n],
        [out.layout.strides[1], out.layout.strides[2]],
        out.layout.offset,
    );

    let a_meta = map_layout(&lhs_mat_layout)?;
    let b_meta = map_layout(&rhs_mat_layout)?;
    let c_meta = map_layout(&out_mat_layout)?;

    let a_batch_stride = to_i64(lhs_batch_stride, "lhs batch stride")?;
    let b_batch_stride = to_i64(rhs_batch_stride, "rhs batch stride")?;
    let c_batch_stride = to_i64(out_batch_stride, "out batch stride")?;

    let key = PipelineKey::BatchedMatmul {
        marker: std::any::TypeId::of::<BatchedMatmulKernel<T>>(),
        scalar: std::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "batched_matmul_kernel", || {
        batched_matmul_shader_source::<T>()
    })?;

    let workgroups_x = n.div_ceil(16) as u32;
    let workgroups_y = m.div_ceil(16) as u32;

    let mut a_ptr = lhs.buffer.raw();
    let mut b_ptr = rhs.buffer.raw();
    let mut c_ptr = out.buffer.raw();
    let mut a_meta_val = a_meta;
    let mut b_meta_val = b_meta;
    let mut c_meta_val = c_meta;
    let mut a_batch_stride_val = a_batch_stride;
    let mut b_batch_stride_val = b_batch_stride;
    let mut c_batch_stride_val = c_batch_stride;

    let mut batch_offset = 0usize;
    while batch_offset < batch {
        let chunk = (batch - batch_offset).min(MAX_GRID_Z);
        let mut batch_offset_val = batch_offset as u32;

        // Argument list mirrors `batched_matmul_kernel(const T*, const T*, T*,
        // MatrixLayout, MatrixLayout, MatrixLayout, long long, long long,
        // long long, unsigned int)`.
        let mut args: [*mut std::ffi::c_void; 10] = [
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut c_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut a_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
            &mut b_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
            &mut c_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
            &mut a_batch_stride_val as *mut i64 as *mut std::ffi::c_void,
            &mut b_batch_stride_val as *mut i64 as *mut std::ffi::c_void,
            &mut c_batch_stride_val as *mut i64 as *mut std::ffi::c_void,
            &mut batch_offset_val as *mut u32 as *mut std::ffi::c_void,
        ];

        launch_kernel(
            device,
            &kernel,
            LaunchConfig::batched_planar(workgroups_x, workgroups_y, chunk as u32, 16, 16),
            &mut args,
        )?;

        batch_offset += chunk;
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
