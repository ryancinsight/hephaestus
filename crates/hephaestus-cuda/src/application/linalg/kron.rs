//! Kronecker product operation on the CUDA device.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};
use leto::Layout;

use super::{map_layout, map_layout_err};
use crate::application::pipeline::{cached_kernel, launch_kernel, LaunchConfig, PipelineKey};
use crate::application::strided::StridedOperand;
use crate::{CudaBuffer, CudaDevice};

struct KronKernel<T>(PhantomData<T>);

fn kron_shader_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
    unsigned int _pad[3];
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
        ty = T::TYPE_TOKEN
    )
}

/// Perform the Kronecker product `out = lhs ⊗ rhs` on the CUDA device.
///
/// For `lhs` with shape `[m, n]` and `rhs` with shape `[p, q]`, the output
/// shape must be `[m * p, n * q]`.
pub fn kron_into<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [lhs_rows, lhs_cols] = lhs.layout.shape;
    let [rhs_rows, rhs_cols] = rhs.layout.shape;
    let expected_rows =
        lhs_rows
            .checked_mul(rhs_rows)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("Kronecker row count overflows usize: {lhs_rows} * {rhs_rows}"),
            })?;
    let expected_cols =
        lhs_cols
            .checked_mul(rhs_cols)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("Kronecker column count overflows usize: {lhs_cols} * {rhs_cols}"),
            })?;

    if out.layout.shape != [expected_rows, expected_cols] {
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

    if expected_rows == 0 || expected_cols == 0 {
        return Ok(());
    }

    let a_meta = map_layout(lhs.layout)?;
    let b_meta = map_layout(rhs.layout)?;
    let out_meta = map_layout(out.layout)?;

    let key = PipelineKey::Kron {
        marker: std::any::TypeId::of::<KronKernel<T>>(),
        scalar: std::any::TypeId::of::<T>(),
    };

    let kernel = cached_kernel(device, key, "kron_kernel", || kron_shader_source::<T>())?;

    let workgroups_x = expected_cols.div_ceil(16);
    let workgroups_y = expected_rows.div_ceil(16);

    let mut a_ptr = lhs.buffer.raw();
    let mut b_ptr = rhs.buffer.raw();
    let mut out_ptr = out.buffer.raw();
    let mut a_meta_val = a_meta;
    let mut b_meta_val = b_meta;
    let mut out_meta_val = out_meta;

    // Argument list mirrors `kron_kernel(const T*, const T*, T*, MatrixLayout,
    // MatrixLayout, MatrixLayout)`.
    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut a_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
        &mut b_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
        &mut out_meta_val as *mut super::GpuMatrixLayout as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(workgroups_x as u32, workgroups_y as u32, 16, 16),
        &mut args,
    )
}

/// Allocate and compute the Kronecker product `lhs ⊗ rhs` on the CUDA device.
///
/// For `lhs` with shape `[m, n]` and `rhs` with shape `[p, q]`, the returned
/// buffer has C-contiguous shape `[m * p, n * q]`.
pub fn kron<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [lhs_rows, lhs_cols] = lhs.layout.shape;
    let [rhs_rows, rhs_cols] = rhs.layout.shape;
    let expected_rows =
        lhs_rows
            .checked_mul(rhs_rows)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("Kronecker row count overflows usize: {lhs_rows} * {rhs_rows}"),
            })?;
    let expected_cols =
        lhs_cols
            .checked_mul(rhs_cols)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("Kronecker column count overflows usize: {lhs_cols} * {rhs_cols}"),
            })?;

    let out_layout =
        Layout::c_contiguous([expected_rows, expected_cols]).map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(out_layout.checked_size().map_err(map_layout_err)?)?;
    kron_into(
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
