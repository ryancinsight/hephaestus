use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
};
use hephaestus_wgpu as wgpu_backend;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

use super::prepared::{PreparedAxisReduction, PreparedReduction};
use super::{MaxOp, MinOp, SumOp};

/// Prepare a generic Metal rank-2 axis reduction into fixed output storage.
///
/// # Errors
///
/// Returns a typed validation or pipeline-preparation error.
pub fn prepare_reduce_axis_into<'a, Op, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'a, T, 2>,
    axis: usize,
    output: crate::application::strided::StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let inner = wgpu_backend::prepare_reduce_axis_into::<Op, T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(output),
        width,
    )?;
    Ok(PreparedAxisReduction { inner })
}

/// Prepare a Metal rank-2 sum reduction along `axis` into fixed output
/// storage.
///
/// # Errors
///
/// Returns a typed validation or pipeline-preparation error.
pub fn prepare_sum_axis_into<'a, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'a, T, 2>,
    axis: usize,
    output: crate::application::strided::StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    prepare_reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Prepare a Metal rank-2 min reduction along `axis` into fixed output
/// storage.
///
/// # Errors
///
/// Returns a typed validation or pipeline-preparation error.
pub fn prepare_min_axis_into<'a, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'a, T, 2>,
    axis: usize,
    output: crate::application::strided::StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
{
    wgpu_backend::prepare_min_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(output),
        width,
    )
    .map(|inner| PreparedAxisReduction { inner })
}

/// Prepare a Metal rank-2 max reduction along `axis` into fixed output
/// storage.
///
/// # Errors
///
/// Returns a typed validation or pipeline-preparation error.
pub fn prepare_max_axis_into<'a, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'a, T, 2>,
    axis: usize,
    output: crate::application::strided::StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
{
    wgpu_backend::prepare_max_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(output),
        width,
    )
    .map(|inner| PreparedAxisReduction { inner })
}

/// Prepare a Metal rank-2 mean reduction along `axis` into fixed output
/// storage.
///
/// # Errors
///
/// Returns a typed validation or pipeline-preparation error.
pub fn prepare_mean_axis_into<'a, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'a, T, 2>,
    axis: usize,
    output: crate::application::strided::StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    wgpu_backend::prepare_mean_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(output),
        width,
    )
    .map(|inner| PreparedAxisReduction { inner })
}

/// Prepare a scalar reduction with a caller-selected power-of-two block width.
///
/// # Errors
///
/// Returns a typed error when the width, allocation, or native pipeline
/// compilation contract is invalid.
pub fn prepare_reduction_with_width<Op, T>(
    device: &MetalDevice,
    input: &MetalBuffer<T>,
    width: BlockWidth,
) -> Result<PreparedReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let inner =
        wgpu_backend::prepare_reduction_with_width::<Op, T>(&device.inner, &input.inner, width)?;
    Ok(PreparedReduction { inner })
}

/// Prepare a scalar reduction using the default block width.
#[inline]
pub fn prepare_reduction<Op, T>(
    device: &MetalDevice,
    input: &MetalBuffer<T>,
) -> Result<PreparedReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    prepare_reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}
