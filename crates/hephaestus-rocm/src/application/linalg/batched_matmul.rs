//! Tiled rank-3 batched matrix multiplication on ROCm.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result};
use leto::Layout;

use super::{GpuMatrixLayout, map_layout, map_layout_err, to_i64};
use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

struct BatchedMatmulKernel<T>(PhantomData<T>);

/// HIP's grid-z dimension limit matches the CUDA-compatible 65,535 bound.
const MAX_GRID_Z: usize = 65_535;

fn shader_source<T: DialectScalar<HipC>>() -> String {
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
    __shared__ {ty} a_tile[16][16];
    __shared__ {ty} b_tile[16][16];

    unsigned int batch = batch_offset + blockIdx.z;
    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;
    unsigned int rows = a_layout.shape[0];
    unsigned int shared = a_layout.shape[1];
    unsigned int cols = b_layout.shape[1];
    long long a_batch_offset = (long long)batch * a_batch_stride;
    long long b_batch_offset = (long long)batch * b_batch_stride;
    long long c_batch_offset = (long long)batch * c_batch_stride;
    {ty} sum = 0;
    unsigned int tile_count = (shared + 15u) / 16u;

    for (unsigned int tile = 0u; tile < tile_count; tile++) {{
        unsigned int a_col = tile * 16u + local_col;
        if (row < rows && a_col < shared) {{
            long long offset = a_batch_offset + (long long)a_layout.offset
                + (long long)row * a_layout.strides[0]
                + (long long)a_col * a_layout.strides[1];
            a_tile[local_row][local_col] = a[offset];
        }} else {{
            a_tile[local_row][local_col] = 0;
        }}

        unsigned int b_row = tile * 16u + local_row;
        if (b_row < shared && col < cols) {{
            long long offset = b_batch_offset + (long long)b_layout.offset
                + (long long)b_row * b_layout.strides[0]
                + (long long)col * b_layout.strides[1];
            b_tile[local_row][local_col] = b[offset];
        }} else {{
            b_tile[local_row][local_col] = 0;
        }}

        __syncthreads();
        for (unsigned int index = 0u; index < 16u; index++) {{
            sum += a_tile[local_row][index] * b_tile[index][local_col];
        }}
        __syncthreads();
    }}

    if (row < rows && col < cols) {{
        long long offset = c_batch_offset + (long long)c_layout.offset
            + (long long)row * c_layout.strides[0]
            + (long long)col * c_layout.strides[1];
        c[offset] = sum;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

/// Compute broadcasted batched matrix products into a caller-owned buffer.
///
/// Singleton input batches broadcast to the output batch count. Each HIP
/// block owns one 16×16 output tile for one batch, and launches are chunked at
/// the grid-z limit rather than serializing one launch per batch.
///
/// # Errors
///
/// Returns a typed dispatch error when batch shapes, layouts, storage, or
/// output aliasing violate the contract, or when HIP module compilation or
/// launch fails.
pub fn batched_matmul_into<T>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
    out: StridedOperand<'_, T, 3>,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod,
{
    let [lhs_batch, rows, lhs_shared] = lhs.layout.shape;
    let [rhs_batch, rhs_shared, cols] = rhs.layout.shape;
    let [out_batch, out_rows, out_cols] = out.layout.shape;
    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok
        || !rhs_batches_ok
        || lhs_shared != rhs_shared
        || rows != out_rows
        || cols != out_cols
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
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
            message: "batched matmul output layout must not contain zero-stride aliasing"
                .to_string(),
        });
    }
    if batch == 0 || rows == 0 || cols == 0 || lhs_shared == 0 {
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
    let lhs_matrix_layout = Layout::new(
        [rows, lhs_shared],
        [lhs.layout.strides[1], lhs.layout.strides[2]],
        lhs.layout.offset,
    );
    let rhs_matrix_layout = Layout::new(
        [rhs_shared, cols],
        [rhs.layout.strides[1], rhs.layout.strides[2]],
        rhs.layout.offset,
    );
    let out_matrix_layout = Layout::new(
        [out_rows, out_cols],
        [out.layout.strides[1], out.layout.strides[2]],
        out.layout.offset,
    );
    let a_meta = map_layout(&lhs_matrix_layout)?;
    let b_meta = map_layout(&rhs_matrix_layout)?;
    let c_meta = map_layout(&out_matrix_layout)?;
    let a_batch_stride = to_i64(lhs_batch_stride, "lhs batch stride")?;
    let b_batch_stride = to_i64(rhs_batch_stride, "rhs batch stride")?;
    let c_batch_stride = to_i64(out_batch_stride, "out batch stride")?;
    let key = PipelineKey::BatchedMatmul {
        marker: core::any::TypeId::of::<BatchedMatmulKernel<T>>(),
        scalar: core::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "batched_matmul_kernel", shader_source::<T>)?;
    let workgroups_x =
        u32::try_from(cols.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: "batched matmul column workgroup count exceeds u32 range".to_string(),
        })?;
    let workgroups_y =
        u32::try_from(rows.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: "batched matmul row workgroup count exceeds u32 range".to_string(),
        })?;

    let mut a_ptr: DevicePtr = lhs.buffer.raw();
    let mut b_ptr: DevicePtr = rhs.buffer.raw();
    let mut c_ptr: DevicePtr = out.buffer.raw();
    let mut a_meta = a_meta;
    let mut b_meta = b_meta;
    let mut c_meta = c_meta;
    let mut a_batch_stride = a_batch_stride;
    let mut b_batch_stride = b_batch_stride;
    let mut c_batch_stride = c_batch_stride;
    let mut batch_offset = 0_usize;
    while batch_offset < batch {
        let chunk = (batch - batch_offset).min(MAX_GRID_Z);
        let mut batch_offset_value =
            u32::try_from(batch_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "batched matmul batch offset exceeds u32 range".to_string(),
            })?;
        let mut args: [*mut core::ffi::c_void; 10] = [
            (&mut a_ptr as *mut DevicePtr).cast(),
            (&mut b_ptr as *mut DevicePtr).cast(),
            (&mut c_ptr as *mut DevicePtr).cast(),
            (&mut a_meta as *mut GpuMatrixLayout).cast(),
            (&mut b_meta as *mut GpuMatrixLayout).cast(),
            (&mut c_meta as *mut GpuMatrixLayout).cast(),
            (&mut a_batch_stride as *mut i64).cast(),
            (&mut b_batch_stride as *mut i64).cast(),
            (&mut c_batch_stride as *mut i64).cast(),
            (&mut batch_offset_value as *mut u32).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::batched_planar(
                workgroups_x,
                workgroups_y,
                u32::try_from(chunk).map_err(|_| HephaestusError::DispatchFailed {
                    message: "batched matmul chunk exceeds u32 range".to_string(),
                })?,
                16,
                16,
            ),
            &mut args,
        )?;
        batch_offset =
            batch_offset
                .checked_add(chunk)
                .ok_or_else(|| HephaestusError::DispatchFailed {
                    message: "batched matmul batch offset overflows usize".to_string(),
                })?;
    }
    Ok(())
}

/// Allocate a C-contiguous output and compute broadcasted batched products.
///
/// Singleton input batches broadcast to the larger batch count. The returned
/// buffer has shape `[batch, lhs.rows, rhs.cols]`.
///
/// # Errors
///
/// Returns a typed dispatch, layout, allocation, module-compilation, or
/// launch error when the operands cannot be multiplied or the device rejects
/// the operation.
pub fn batched_matmul<T>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod,
{
    let [lhs_batch, rows, lhs_shared] = lhs.layout.shape;
    let [rhs_batch, rhs_shared, cols] = rhs.layout.shape;
    let batch = lhs_batch.max(rhs_batch);
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_shared != rhs_shared {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape, rhs.layout.shape
            ),
        });
    }
    let output_layout = Layout::c_contiguous([batch, rows, cols]).map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    batched_matmul_into(
        device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::shader_source;

    #[test]
    fn source_declares_batch_dispatch_contract() {
        let source = shader_source::<i32>();
        assert!(source.contains("__shared__ int a_tile[16][16];"));
        assert!(source.contains("blockIdx.z"));
        assert!(source.contains("batch_offset"));
        assert!(source.contains("long long a_batch_stride"));
    }
}
