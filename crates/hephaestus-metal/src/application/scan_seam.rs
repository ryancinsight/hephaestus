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
    T: DialectScalar<Wgsl> + eunomia::Pod + Send + Sync,
{
    type Dialect = Wgsl;
    type PreparedScan<'op, const N: usize>
        = PreparedScan
    where
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
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
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        <WgpuScanOps as ScanOps<WgpuDevice, T>>::dispatch_scan::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
