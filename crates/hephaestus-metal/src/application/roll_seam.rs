//! Provider-owned n-D circular-shift seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, RollOps, StridedView, Wgsl};
use hephaestus_wgpu::WgpuRollOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`RollOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalRollOps {
    inner: WgpuRollOps,
}

impl<T> RollOps<MetalDevice, T> for MetalRollOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn roll_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        shifts: &[isize],
        dims: &[usize],
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.roll_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            shifts,
            dims,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
