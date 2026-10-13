use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CudaC, DialectScalar, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, Result,
    SumOp,
};

use crate::CudaDevice;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;

use super::axis::{reduce_axis, reduce_axis_into};
use super::plan::{axis_len, reject_empty_axis};

/// Sum-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn sum_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Sum-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn sum_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    reduce_axis::<SumOp, T>(device, input, axis, width)
}

/// Product-reduce a rank-2 strided matrix along `axis`, preserving the reduced
/// axis as length one.
#[inline]
pub fn prod_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<ProdOp> + IdentityToken<ProdOp, CudaC>,
{
    reduce_axis_into::<ProdOp, T>(device, input, axis, output, width)
}

/// Product-reduce a rank-2 strided matrix along `axis`, allocating a
/// C-contiguous output buffer.
#[inline]
pub fn prod_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<ProdOp> + IdentityToken<ProdOp, CudaC>,
{
    reduce_axis::<ProdOp, T>(device, input, axis, width)
}

/// Min-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn min_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Min-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn min_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis::<MinOp, T>(device, input, axis, width)
}

/// Max-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn max_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Max-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn max_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis::<MaxOp, T>(device, input, axis, width)
}
