//! Tiled rank-2 matrix multiplication on ROCm.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result};
use leto::Layout;

use super::{GpuMatrixLayout, map_layout, map_layout_err};
use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

struct MatmulKernel<T>(PhantomData<T>);

fn shader_source<T: DialectScalar<HipC>>() -> String {
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
    __shared__ {ty} a_tile[16][16];
    __shared__ {ty} b_tile[16][16];

    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;
    unsigned int rows = a_layout.shape[0];
    unsigned int shared = a_layout.shape[1];
    unsigned int cols = b_layout.shape[1];
    {ty} sum = 0;
    unsigned int tile_count = (shared + 15u) / 16u;

    for (unsigned int tile = 0u; tile < tile_count; tile++) {{
        unsigned int a_col = tile * 16u + local_col;
        if (row < rows && a_col < shared) {{
            int offset = (int)a_layout.offset + (int)row * a_layout.strides[0]
                + (int)a_col * a_layout.strides[1];
            a_tile[local_row][local_col] = a[offset];
        }} else {{
            a_tile[local_row][local_col] = 0;
        }}

        unsigned int b_row = tile * 16u + local_row;
        if (b_row < shared && col < cols) {{
            int offset = (int)b_layout.offset + (int)b_row * b_layout.strides[0]
                + (int)col * b_layout.strides[1];
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
        int offset = (int)c_layout.offset + (int)row * c_layout.strides[0]
            + (int)col * c_layout.strides[1];
        c[offset] = sum;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

/// Compute `out = lhs * rhs` over rank-2 strided device operands.
///
/// The output may be strided but may not alias either input or contain
/// zero-stride aliasing. The contracted extent is tiled in 16-element chunks;
/// partial edge tiles are zero-filled before the shared-memory multiply.
///
/// # Errors
///
/// Returns a typed dispatch error when shapes, layouts, storage bounds, or
/// aliasing violate the matrix multiplication contract, or when HIP module
/// compilation or launch fails.
pub fn matmul_into<T>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod,
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
        marker: core::any::TypeId::of::<MatmulKernel<T>>(),
        scalar: core::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "matmul_kernel", shader_source::<T>)?;
    let workgroups_x =
        u32::try_from(cols.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: "matmul column workgroup count exceeds u32 range".to_string(),
        })?;
    let workgroups_y =
        u32::try_from(rows.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: "matmul row workgroup count exceeds u32 range".to_string(),
        })?;

    let mut a_ptr: DevicePtr = lhs.buffer.raw();
    let mut b_ptr: DevicePtr = rhs.buffer.raw();
    let mut c_ptr: DevicePtr = out.buffer.raw();
    let mut a_meta = a_meta;
    let mut b_meta = b_meta;
    let mut c_meta = c_meta;
    let mut args: [*mut core::ffi::c_void; 6] = [
        (&mut a_ptr as *mut DevicePtr).cast(),
        (&mut b_ptr as *mut DevicePtr).cast(),
        (&mut c_ptr as *mut DevicePtr).cast(),
        (&mut a_meta as *mut GpuMatrixLayout).cast(),
        (&mut b_meta as *mut GpuMatrixLayout).cast(),
        (&mut c_meta as *mut GpuMatrixLayout).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(workgroups_x, workgroups_y, 16, 16),
        &mut args,
    )
}

/// Allocate a C-contiguous output and compute `lhs * rhs` on ROCm.
///
/// # Errors
///
/// Returns a typed dispatch, layout, allocation, module-compilation, or
/// launch error when the operands cannot be multiplied or the device rejects
/// the operation.
pub fn matmul<T>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod,
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
    let output_layout = Layout::c_contiguous([rows, cols]).map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    matmul_into(
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
    fn source_declares_tiled_shared_memory_contract() {
        let source = shader_source::<i32>();
        assert!(source.contains("__shared__ int a_tile[16][16];"));
        assert!(source.contains("__shared__ int b_tile[16][16];"));
        assert!(source.contains("blockIdx.x"));
        assert!(source.contains("blockIdx.y"));
        assert!(source.contains("__syncthreads();"));
    }
}
