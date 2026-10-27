//! Public CUDA dot, trace, and norm entry points.

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DialectScalar, HephaestusError, IdentityToken, MaxOp,
    OpIdentity, Result, SumOp,
};
use leto::Layout;

use super::plan::map_reduction;
use super::shader::{AbsMap, DotMap, IdentityMap, MaxAbsMap, SquareMap};
use crate::CudaDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;

/// Compute the vector dot product `Σᵢ a[i] * b[i]` on the CUDA device.
pub fn dot<T>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    if a.layout.shape() != b.layout.shape() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                a.layout.shape(),
                b.layout.shape()
            ),
        });
    }

    map_reduction::<DotMap, T, 1>(device, a, b)
}

/// Compute the trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix on the CUDA device.
pub fn trace<T>(device: &CudaDevice, matrix: StridedOperand<'_, T, 2>) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    let [rows, cols] = matrix.layout.shape();
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "trace requires a square matrix, got shape {:?}",
                matrix.layout.shape()
            ),
        });
    }

    if rows == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let diagonal_stride = matrix.layout.strides()[0]
        .checked_add(matrix.layout.strides()[1])
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: "trace diagonal stride overflows isize".to_string(),
        })?;
    let diag_layout = Layout::try_new([rows], [diagonal_stride], matrix.layout.offset())
        .expect("invariant: submatrix layout derives from a validated parent");
    let diag_operand = StridedOperand {
        buffer: matrix.buffer,
        layout: &diag_layout,
    };

    map_reduction::<IdentityMap, T, 1>(device, diag_operand, diag_operand)
}

/// Compute the L1 norm `Σ |x|` on the CUDA device.
pub fn norm_l1<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    map_reduction::<AbsMap, T, N>(device, view, view)
}

/// Compute the L2 / Frobenius norm `sqrt(Σ x²)` on the CUDA device.
pub fn norm_l2<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    let squared_sum = map_reduction::<SquareMap, T, N>(device, view, view)?;
    let out = device.alloc_uninitialized::<T>(1)?;
    unary_elementwise_into::<SqrtOp, T>(device, &squared_sum, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}

/// Compute the Max norm `max |x|` on the CUDA device.
pub fn norm_max<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
{
    map_reduction::<MaxAbsMap, T, N>(device, view, view)
}
