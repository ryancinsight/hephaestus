//! Matrix power operation on the CUDA device.

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result,
};
use leto::Layout;

use super::{map_layout_err, matmul_into};
use crate::application::strided::{unary_elementwise_strided_into, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

/// CUDA scalar whose host identity values support matrix-power initialization.
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

fn identity_matrix<T: MatrixIdentityScalar>(n: usize) -> Vec<T> {
    let mut values = vec![T::ZERO; n * n];
    for i in 0..n {
        values[i * n + i] = T::ONE;
    }
    values
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
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matpow requires a square matrix, got shape {:?}",
                matrix.layout.shape
            ),
        });
    }

    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let layout = Layout::c_contiguous([rows, rows]).map_err(map_layout_err)?;
    let mut result = device.upload(&identity_matrix::<T>(rows))?;
    if exponent == 0 {
        return Ok(result);
    }

    let mut base = device.alloc_zeroed::<T>(rows * rows)?;
    unary_elementwise_strided_into::<crate::application::elementwise::IdentityOp, T, 2>(
        device,
        matrix,
        StridedOperand {
            buffer: &base,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )?;

    let mut result_scratch = device.alloc_zeroed::<T>(rows * rows)?;
    let mut base_scratch = device.alloc_zeroed::<T>(rows * rows)?;
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
