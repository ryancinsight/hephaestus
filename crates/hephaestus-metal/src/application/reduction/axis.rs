use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
};
use hephaestus_wgpu as wgpu_backend;
use hephaestus_wgpu::{MaxOp, MinOp, SumOp};

/// A reusable Metal rank-2 axis reduction plan delegated to WGPU's native
/// Metal path.
pub struct PreparedAxisReduction<T> {
    inner: wgpu_backend::PreparedAxisReduction<T>,
}

impl<T> PreparedAxisReduction<T> {
    /// Dispatch the prepared axis reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when command encoding or submission
    /// fails on the Metal-selected device.
    pub fn dispatch(&self, device: &MetalDevice) -> Result<()> {
        self.inner.dispatch(&device.inner)
    }
}

/// Submit several prepared Metal axis reductions in one native WGPU command
/// batch.
///
/// # Errors
///
/// Returns a typed dispatch error when command encoding or submission fails.
pub fn submit_prepared_axis_reduction_batch<T>(
    device: &MetalDevice,
    reductions: &[&PreparedAxisReduction<T>],
) -> Result<()> {
    let inner = reductions
        .iter()
        .map(|reduction| &reduction.inner)
        .collect::<Vec<_>>();
    wgpu_backend::submit_prepared_axis_reduction_batch(&device.inner, &inner)
}

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
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    hephaestus_wgpu::reduce_axis_into::<Op, T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}
