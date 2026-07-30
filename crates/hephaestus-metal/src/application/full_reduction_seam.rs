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
    T: DialectScalar<Wgsl> + bytemuck::Pod + Send + Sync,
{
    type Dialect = Wgsl;
    type Prepared<const N: usize> = PreparedFullReduction<T>;

    fn prepare_reduce_full<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        output: StridedView<'_, MetalBuffer<T>, 1>,
    ) -> Result<Self::Prepared<N>>
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
        prepared: &Self::Prepared<N>,
    ) -> Result<()> {
        <WgpuFullReductionOps as FullReductionOps<WgpuDevice, T>>::dispatch_full::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
