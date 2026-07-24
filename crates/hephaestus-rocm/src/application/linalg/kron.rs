//! Kronecker products over strided ROCm matrix operands.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result};
use leto::Layout;

use super::{GpuMatrixLayout, map_layout, map_layout_err};
use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

struct KronKernel<T>(PhantomData<T>);

fn shader_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
}};

extern "C" __global__ void kron_kernel(
    const {ty}* a,
    const {ty}* b,
    {ty}* out,
    MatrixLayout a_layout,
    MatrixLayout b_layout,
    MatrixLayout out_layout
) {{
    unsigned int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int b_rows = b_layout.shape[0];
    unsigned int b_cols = b_layout.shape[1];
    unsigned int rows = a_layout.shape[0] * b_rows;
    unsigned int cols = a_layout.shape[1] * b_cols;

    if (out_row >= rows || out_col >= cols) {{
        return;
    }}

    unsigned int a_row = out_row / b_rows;
    unsigned int a_col = out_col / b_cols;
    unsigned int b_row = out_row % b_rows;
    unsigned int b_col = out_col % b_cols;

    int a_offset = (int)a_layout.offset
        + (int)a_row * a_layout.strides[0]
        + (int)a_col * a_layout.strides[1];
    int b_offset = (int)b_layout.offset
        + (int)b_row * b_layout.strides[0]
        + (int)b_col * b_layout.strides[1];
    int out_offset = (int)out_layout.offset
        + (int)out_row * out_layout.strides[0]
        + (int)out_col * out_layout.strides[1];

    out[out_offset] = a[a_offset] * b[b_offset];
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

fn output_shape(lhs: &Layout<2>, rhs: &Layout<2>) -> Result<[usize; 2]> {
    let rows =
        lhs.shape[0]
            .checked_mul(rhs.shape[0])
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!(
                    "Kronecker row count overflows usize: {} * {}",
                    lhs.shape[0], rhs.shape[0]
                ),
            })?;
    let cols =
        lhs.shape[1]
            .checked_mul(rhs.shape[1])
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!(
                    "Kronecker column count overflows usize: {} * {}",
                    lhs.shape[1], rhs.shape[1]
                ),
            })?;
    Ok([rows, cols])
}

/// Compute `out = lhs ⊗ rhs` over rank-2 strided ROCm operands.
///
/// For `lhs` with shape `[m, n]` and `rhs` with shape `[p, q]`, the output
/// shape is `[m * p, n * q]`. The HIP kernel maps each output coordinate back
/// to its source coordinates, so non-contiguous input and output views retain
/// their Leto layout semantics.
///
/// # Errors
///
/// Returns a typed dispatch error when shapes, layouts, storage bounds, or
/// aliasing violate the contract, or when HIP module compilation or launch
/// fails.
pub fn kron_into<T>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod,
{
    let expected_shape = output_shape(lhs.layout, rhs.layout)?;
    if out.layout.shape != expected_shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Kronecker output shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
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
            message: "Kronecker output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    let [rows, cols] = expected_shape;
    if rows == 0 || cols == 0 {
        return Ok(());
    }

    let a_meta = map_layout(lhs.layout)?;
    let b_meta = map_layout(rhs.layout)?;
    let out_meta = map_layout(out.layout)?;
    let key = PipelineKey::Kron {
        marker: core::any::TypeId::of::<KronKernel<T>>(),
        scalar: core::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "kron_kernel", shader_source::<T>)?;
    let workgroups_x =
        u32::try_from(cols.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: "Kronecker column workgroup count exceeds u32 range".to_string(),
        })?;
    let workgroups_y =
        u32::try_from(rows.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: "Kronecker row workgroup count exceeds u32 range".to_string(),
        })?;

    let mut a_ptr: DevicePtr = lhs.buffer.raw();
    let mut b_ptr: DevicePtr = rhs.buffer.raw();
    let mut out_ptr: DevicePtr = out.buffer.raw();
    let mut a_meta = a_meta;
    let mut b_meta = b_meta;
    let mut out_meta = out_meta;
    let mut args: [*mut core::ffi::c_void; 6] = [
        (&mut a_ptr as *mut DevicePtr).cast(),
        (&mut b_ptr as *mut DevicePtr).cast(),
        (&mut out_ptr as *mut DevicePtr).cast(),
        (&mut a_meta as *mut GpuMatrixLayout).cast(),
        (&mut b_meta as *mut GpuMatrixLayout).cast(),
        (&mut out_meta as *mut GpuMatrixLayout).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(workgroups_x, workgroups_y, 16, 16),
        &mut args,
    )
}

/// Allocate a C-contiguous output and compute `lhs ⊗ rhs` on ROCm.
///
/// The returned buffer has shape `[lhs.rows * rhs.rows, lhs.cols * rhs.cols]`.
///
/// # Errors
///
/// Returns a typed dispatch, layout, allocation, module-compilation, or launch
/// error when the operands cannot be combined or the device rejects the
/// operation.
pub fn kron<T>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod,
{
    let shape = output_shape(lhs.layout, rhs.layout)?;
    let output_layout = Layout::c_contiguous(shape).map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    kron_into(
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
    fn source_declares_strided_coordinate_mapping_contract() {
        let source = shader_source::<i32>();
        assert!(source.contains("out_row / b_rows"));
        assert!(source.contains("out_col % b_cols"));
        assert!(source.contains("a_layout.strides[0]"));
        assert!(source.contains("out_layout.strides[1]"));
        assert!(source.contains("out[out_offset] = a[a_offset] * b[b_offset]"));
    }
}
