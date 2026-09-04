//! Matrix power operation on the CUDA device.

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result,
};
use leto::Layout;

use super::{map_layout_err, matmul_into};
use crate::CudaDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::{StridedOperand, unary_elementwise_strided_into};
use crate::infrastructure::buffer::CudaBuffer;

/// CUDA scalar supporting matrix identity initialization.
pub trait MatrixIdentityScalar: DialectScalar<CudaC> + Pod {
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
}

impl MatrixIdentityScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
impl MatrixIdentityScalar for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}
impl MatrixIdentityScalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

fn identity_shader_source<T: MatrixIdentityScalar>() -> String {
    format!(
        r#"
extern "C" __global__ void matrix_identity_kernel(
    {ty}* out,
    unsigned int rows,
    {ty} zero,
    {ty} one
) {{
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && column < rows) {{
        size_t index = (size_t)row * (size_t)rows + (size_t)column;
        out[index] = row == column ? one : zero;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

pub(crate) fn device_identity<T>(
    device: &CudaDevice,
    rows: usize,
    len: usize,
) -> Result<CudaBuffer<T>>
where
    T: MatrixIdentityScalar,
{
    let output = device.alloc_uninitialized::<T>(len)?;
    if len == 0 {
        return Ok(output);
    }
    let kernel = cached_kernel(
        device,
        PipelineKey::MatrixIdentity {
            scalar: core::any::TypeId::of::<T>(),
        },
        "matrix_identity_kernel",
        identity_shader_source::<T>,
    )?;
    let mut output_ptr = output.raw();
    let mut rows = u32::try_from(rows).map_err(|_| HephaestusError::DispatchFailed {
        message: "matpow row count exceeds u32 range".to_string(),
    })?;
    let mut zero = T::ZERO;
    let mut one = T::ONE;
    let mut args: [*mut core::ffi::c_void; 4] = [
        (&mut output_ptr as *mut u64).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut zero as *mut T).cast(),
        (&mut one as *mut T).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(rows.div_ceil(16), rows.div_ceil(16), 16, 16),
        &mut args,
    )?;
    Ok(output)
}

/// Raise a square matrix to a non-negative integer power on the CUDA device.
///
/// The algorithm is exponentiation by squaring, matching Leto's `matpow`
/// contract: `A^0` is the identity matrix and non-square inputs are rejected.
/// Matrix products are dispatched through [`matmul_into`].
pub fn matpow<T>(
    device: &CudaDevice,
    matrix: StridedOperand<'_, T, 2>,
    exponent: u32,
) -> Result<CudaBuffer<T>>
where
    T: MatrixIdentityScalar,
{
    let [rows, cols] = matrix.layout.shape();
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matpow requires a square matrix, got shape {:?}",
                matrix.layout.shape()
            ),
        });
    }

    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let layout = Layout::c_contiguous([rows, rows]).map_err(map_layout_err)?;
    let n_sq = layout.checked_size().map_err(map_layout_err)?;
    let mut result = device_identity::<T>(device, rows, n_sq)?;
    if exponent == 0 {
        return Ok(result);
    }

    let mut base = device.alloc_uninitialized::<T>(n_sq)?;
    unary_elementwise_strided_into::<crate::application::elementwise::IdentityOp, T, 2>(
        device,
        matrix,
        StridedOperand {
            buffer: &base,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )?;

    let mut result_scratch = device.alloc_uninitialized::<T>(n_sq)?;
    let mut base_scratch = device.alloc_uninitialized::<T>(n_sq)?;
    let mut remaining = exponent;

    loop {
        if remaining & 1 == 1 {
            matmul_into(
                device,
                StridedOperand {
                    buffer: &result,
                    layout: &layout,
                },
                StridedOperand {
                    buffer: &base,
                    layout: &layout,
                },
                StridedOperand {
                    buffer: &result_scratch,
                    layout: &layout,
                },
            )?;
            core::mem::swap(&mut result, &mut result_scratch);
        }

        remaining >>= 1;
        if remaining == 0 {
            break;
        }

        matmul_into(
            device,
            StridedOperand {
                buffer: &base,
                layout: &layout,
            },
            StridedOperand {
                buffer: &base,
                layout: &layout,
            },
            StridedOperand {
                buffer: &base_scratch,
                layout: &layout,
            },
        )?;
        core::mem::swap(&mut base, &mut base_scratch);
    }

    Ok(result)
}
