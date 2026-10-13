//! Integer matrix powers.

use super::identity::device_identity;
use super::matmul::matmul_into;
use super::scalars::MatrixIdentityScalar;
use crate::application::strided::{StridedOperand, map_layout_err, unary_elementwise_strided_into};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl};
use leto::Layout;

/// Raise a square matrix to a non-negative integer power on the GPU.
///
/// The algorithm is exponentiation by squaring, matching Leto's `matpow`
/// contract: `A^0` is the identity matrix and non-square inputs are rejected.
/// Matrix products are dispatched through [`matmul_into`]; the host controls only
/// the exponent bits and buffer rotation.
pub fn matpow<T>(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, T, 2>,
    exponent: u32,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + MatrixIdentityScalar,
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
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    let layout = Layout::c_contiguous([rows, rows]).map_err(map_layout_err)?;
    let n_sq = rows
        .checked_mul(rows)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!(
                "matpow: matrix size {rows}\u{00d7}{rows} overflows usize ({rows}^2 > usize::MAX)"
            ),
        })?;
    let mut result = device_identity::<T>(device, &layout)?;
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
