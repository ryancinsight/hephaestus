use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
};
use hephaestus_wgpu as wgpu_backend;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

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
