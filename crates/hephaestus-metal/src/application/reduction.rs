//! Reduction operations.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{BlockWidth, Result};
use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::{MaxOp, MinOp, SumOp};

/// Run reduction on the device, returning a 1-element buffer holding the result.
#[inline]
pub fn reduction<Op, T>(device: &MetalDevice, buffer: &MetalBuffer<T>) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::ReductionWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<Op>,
{
    let inner = wgpu_backend::reduction::<Op, T>(&device.inner, &buffer.inner)?;
    Ok(MetalBuffer { inner })
}

/// Run reduction on the device with a caller-selected power-of-two block width.
#[inline]
pub fn reduction_with_width<Op, T>(
    device: &MetalDevice,
    buffer: &MetalBuffer<T>,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::ReductionWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<Op>,
{
    let inner = wgpu_backend::reduction_with_width::<Op, T>(&device.inner, &buffer.inner, width)?;
    Ok(MetalBuffer { inner })
}

/// Reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn reduce_axis<Op, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::ReductionWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<Op>,
{
    let inner = wgpu_backend::reduce_axis::<Op, T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn reduce_axis_into<Op, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: wgpu_backend::ReductionWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<Op>,
{
    hephaestus_wgpu::reduce_axis_into::<Op, T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}

/// Sum-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn sum_axis<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<SumOp>,
{
    let inner = wgpu_backend::sum_axis::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Sum-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn sum_axis_into<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<SumOp>,
{
    wgpu_backend::sum_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}

/// Min-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn min_axis<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<MinOp>,
{
    let inner = wgpu_backend::min_axis::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Min-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn min_axis_into<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<MinOp>,
{
    wgpu_backend::min_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}

/// Max-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn max_axis<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<MaxOp>,
{
    let inner = wgpu_backend::max_axis::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Max-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn max_axis_into<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<MaxOp>,
{
    wgpu_backend::max_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}

/// Mean-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn mean_axis<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<SumOp>,
{
    let inner = wgpu_backend::mean_axis::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Mean-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn mean_axis_into<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ReductionIdentity<SumOp>,
{
    wgpu_backend::mean_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}
