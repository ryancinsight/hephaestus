//! Provider-owned scan seam for Metal.

use hephaestus_core::{
    CombineExpr, DialectScalar, IdentityToken, OpIdentity, Result, ScanDirection, ScanOps,
    StridedView, Wgsl,
};
use hephaestus_wgpu::{PreparedScan, WgpuDevice, WgpuScanOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`ScanOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalScanOps {
    inner: WgpuScanOps,
}

impl<T> ScanOps<MetalDevice, T> for MetalScanOps
where
    T: DialectScalar<Wgsl> + bytemuck::Pod + Send + Sync,
{
    type Dialect = Wgsl;
    type PreparedScan<const N: usize> = PreparedScan;

    fn prepare_scan_axis<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        self.inner.prepare_scan_axis::<Op, N>(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            direction,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedScan<N>,
    ) -> Result<()> {
        <WgpuScanOps as ScanOps<WgpuDevice, T>>::dispatch_scan::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
