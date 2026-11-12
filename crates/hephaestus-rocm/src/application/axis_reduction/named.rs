//! Named rank-2 axis reductions over strided operands on ROCm.

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, DialectScalar, HipC, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, Result, SumOp,
};

use crate::RocmDevice;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

use super::axis::{reduce_axis, reduce_axis_into};
use super::plan::{axis_len, reject_empty_axis};

/// Sum-reduce a rank-2 strided operand along `axis` into caller-owned storage.
#[inline]
pub fn sum_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Sum-reduce a rank-2 strided operand along `axis` into a C-contiguous buffer.
#[inline]
pub fn sum_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reduce_axis::<SumOp, T>(device, input, axis, width)
}

/// Product-reduce a rank-2 strided operand along `axis` into caller-owned storage.
#[inline]
pub fn prod_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<ProdOp> + IdentityToken<ProdOp, HipC>,
{
    reduce_axis_into::<ProdOp, T>(device, input, axis, output, width)
}

/// Product-reduce a rank-2 strided operand along `axis` into a C-contiguous buffer.
#[inline]
pub fn prod_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<ProdOp> + IdentityToken<ProdOp, HipC>,
{
    reduce_axis::<ProdOp, T>(device, input, axis, width)
}

/// Min-reduce a rank-2 strided operand along `axis` into caller-owned storage.
#[inline]
pub fn min_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Min-reduce a rank-2 strided operand along `axis` into a C-contiguous buffer.
#[inline]
pub fn min_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis::<MinOp, T>(device, input, axis, width)
}

/// Max-reduce a rank-2 strided operand along `axis` into caller-owned storage.
#[inline]
pub fn max_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Max-reduce a rank-2 strided operand along `axis` into a C-contiguous buffer.
#[inline]
pub fn max_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis::<MaxOp, T>(device, input, axis, width)
}
