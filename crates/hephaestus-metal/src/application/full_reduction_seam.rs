//! Provider-owned full-reduction seam for Metal.

use hephaestus_core::{
    CombineExpr, DialectScalar, FullReductionOps, IdentityToken, OpIdentity, Result, StridedView,
    Wgsl,
};
use hephaestus_wgpu::{PreparedFullReduction, WgpuDevice, WgpuFullReductionOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`FullReductionOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalFullReductionOps {
    inner: WgpuFullReductionOps,
}

impl<T> FullReductionOps<MetalDevice, T> for MetalFullReductionOps
where
    T: DialectScalar<Wgsl> + eunomia::Pod + Send + Sync,
{
    type Dialect = Wgsl;
    type Prepared<'op, const N: usize>
        = PreparedFullReduction<T>
    where
        T: 'op;

    fn prepare_reduce_full<'op, Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<T>, N>,
        output: StridedView<'op, MetalBuffer<T>, 1>,
    ) -> Result<Self::Prepared<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        self.inner.prepare_reduce_full::<Op, N>(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn dispatch_full<const N: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::Prepared<'_, N>,
    ) -> Result<()> {
        <WgpuFullReductionOps as FullReductionOps<WgpuDevice, T>>::dispatch_full::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
