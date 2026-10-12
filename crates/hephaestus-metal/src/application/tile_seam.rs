//! Provider-owned n-D tiling seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, StridedView, TileOps, TileReps, Wgsl};
use hephaestus_wgpu::WgpuTileOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`TileOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalTileOps {
    inner: WgpuTileOps,
}

impl<T> TileOps<MetalDevice, T> for MetalTileOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn tile_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        reps: TileReps<N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.tile_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            reps,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
