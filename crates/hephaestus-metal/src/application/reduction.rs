//! Reduction operations.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
};
use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::{MaxOp, MinOp, SumOp};

/// A reusable Metal scalar reduction plan delegated to WGPU's native Metal path.
pub struct PreparedReduction<T> {
    inner: wgpu_backend::PreparedReduction<T>,
}

impl<T> PreparedReduction<T> {
    /// Dispatch the prepared reduction and reuse its device-resident outputs.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when command encoding or submission
    /// fails on the Metal-selected device.
    pub fn dispatch(&self, device: &MetalDevice) -> Result<()> {
        self.inner.dispatch(&device.inner)
    }

    /// Return a handle to the one-element output buffer holding the latest result.
    #[must_use]
    pub fn output(&self) -> MetalBuffer<T>
    where
        T: Clone,
    {
        MetalBuffer {
            inner: self.inner.output().clone(),
        }
    }
}

/// Submit several prepared Metal reductions in one native WGPU command batch.
///
/// # Errors
///
/// Returns a typed dispatch error when command encoding or submission fails.
pub fn submit_prepared_reduction_batch<Op, T>(
    device: &MetalDevice,
    reductions: &[&PreparedReduction<T>],
) -> Result<()> {
    let inner = reductions
        .iter()
        .map(|reduction| &reduction.inner)
        .collect::<Vec<_>>();
    wgpu_backend::submit_prepared_reduction_batch(&device.inner, &inner)
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    prepare_reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}

/// Run reduction on the device, returning a 1-element buffer holding the result.
#[inline]
pub fn reduction<Op, T>(device: &MetalDevice, buffer: &MetalBuffer<T>) -> Result<MetalBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
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
    T: DialectScalar<Wgsl> + bytemuck::Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    wgpu_backend::mean_axis_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}
