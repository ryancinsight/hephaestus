//! Public ROCm dot, trace, and norm entry points.

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, HipC, IdentityToken, MaxOp,
    OpIdentity, Result, SumOp,
};
use leto::Layout;

use super::plan::map_reduction;
use super::shader::{AbsMap, DotMap, IdentityMap, MaxAbsMap, SquareMap};
use crate::RocmDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// Compute the vector dot product `Σᵢ a[i] * b[i]` on the ROCm device.
///
/// The two rank-1 operands must have equal logical shapes. Layout validation
/// occurs before the HIP launch, and the returned buffer contains one scalar.
pub fn dot<T>(
    device: &RocmDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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

/// Compute the trace tr(A) = Σᵢ aᵢᵢ of a square matrix on ROCm.
pub fn trace<T>(device: &RocmDevice, matrix: StridedOperand<'_, T, 2>) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
    let diagonal_layout = Layout::try_new([rows], [diagonal_stride], matrix.layout.offset())
        .expect("invariant: submatrix layout derives from a validated parent");
    let diagonal = StridedOperand {
        buffer: matrix.buffer,
        layout: &diagonal_layout,
    };
    map_reduction::<IdentityMap, T, 1>(device, diagonal, diagonal)
}

/// Compute the entrywise L1 norm Σ |x| on ROCm.
pub fn norm_l1<T, const N: usize>(
    device: &RocmDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    map_reduction::<AbsMap, T, N>(device, view, view)
}

/// Scalar types supported by ROCm L2/Frobenius norms.
pub trait L2NormScalar:
    DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>
{
}

impl L2NormScalar for f32 {}

/// Compute the L2/Frobenius norm sqrt(Σ x²) on ROCm.
pub fn norm_l2<T, const N: usize>(
    device: &RocmDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    T: L2NormScalar,
{
    let squared_sum = map_reduction::<SquareMap, T, N>(device, view, view)?;
    let output = device.alloc_uninitialized::<T>(1)?;
    unary_elementwise_into::<SqrtOp, T>(device, &squared_sum, &output, BlockWidth::DEFAULT)?;
    Ok(output)
}

/// Compute the max-magnitude norm max |x| on ROCm.
pub fn norm_max<T, const N: usize>(
    device: &RocmDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    map_reduction::<MaxAbsMap, T, N>(device, view, view)
}
