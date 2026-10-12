use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
};
use hephaestus_wgpu as wgpu_backend;

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
pub fn submit_prepared_reduction_batch<T>(
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

/// Run reduction on the device, returning a 1-element buffer holding the result.
#[inline]
pub fn reduction<Op, T>(device: &MetalDevice, buffer: &MetalBuffer<T>) -> Result<MetalBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    T: DialectScalar<Wgsl> + eunomia::Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let inner = wgpu_backend::reduction_with_width::<Op, T>(&device.inner, &buffer.inner, width)?;
    Ok(MetalBuffer { inner })
}
