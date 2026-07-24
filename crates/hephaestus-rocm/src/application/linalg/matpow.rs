//! Matrix powers over ROCm device-resident operands.

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result,
};
use leto::Layout;

use super::{map_layout_err, matmul_into};
use crate::RocmDevice;
use crate::application::elementwise::IdentityOp;
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::RocmBuffer;

/// ROCm scalar whose host identity values support matrix-power initialization.
pub trait MatrixIdentityScalar: DialectScalar<HipC> + Pod {
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

fn identity_matrix<T: MatrixIdentityScalar>(n: usize) -> Result<Vec<T>> {
    let len = n
        .checked_mul(n)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("identity matrix size {n}×{n} overflows usize"),
        })?;
    let mut values = vec![T::ZERO; len];
    for index in 0..n {
        values[index * n + index] = T::ONE;
    }
    Ok(values)
}

/// Raise a square matrix to a non-negative integer power on ROCm.
///
/// The algorithm is exponentiation by squaring, matching Leto's `matpow`
/// contract: `A^0` is the identity matrix and non-square inputs are rejected.
/// The input view is copied into contiguous device storage through the native
/// strided identity kernel, and every product is dispatched through
/// [`matmul_into`].
///
/// # Errors
///
/// Returns a typed dispatch, layout, allocation, transfer, module-compilation,
/// or launch error when the input is invalid or ROCm rejects an operation.
pub fn matpow<T>(
    device: &RocmDevice,
    matrix: StridedOperand<'_, T, 2>,
    exponent: u32,
) -> Result<RocmBuffer<T>>
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
    let n_sq = rows
        .checked_mul(rows)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("matpow matrix size {rows}×{rows} overflows usize"),
        })?;
    let mut result = device.upload(&identity_matrix::<T>(rows)?)?;
    if exponent == 0 {
        return Ok(result);
    }

    let mut base = device.alloc_zeroed::<T>(n_sq)?;
    unary_elementwise_strided_into::<IdentityOp, T, 2>(
        device,
        matrix,
        StridedOperand {
            buffer: &base,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )?;

    let mut result_scratch = device.alloc_zeroed::<T>(n_sq)?;
    let mut base_scratch = device.alloc_zeroed::<T>(n_sq)?;
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

#[cfg(test)]
mod tests {
    use super::identity_matrix;

    #[test]
    fn identity_matrix_places_only_the_multiplicative_identity_on_diagonal() {
        assert_eq!(
            identity_matrix::<i32>(3).expect("valid identity"),
            [1, 0, 0, 0, 1, 0, 0, 0, 1]
        );
    }
}
