//! Provider-owned batched cross-product seam for Metal.

use eunomia::Pod;
use hephaestus_core::{CrossProductOps, DialectScalar, Result, Wgsl};
use hephaestus_wgpu::WgpuCrossProductOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`CrossProductOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalCrossProductOps {
    inner: WgpuCrossProductOps,
}

impl<T> CrossProductOps<MetalDevice, T> for MetalCrossProductOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn cross_into(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer<T>,
        b: &MetalBuffer<T>,
        out: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner
            .cross_into(device.wgpu_device(), &a.inner, &b.inner, &out.inner)
    }
}
