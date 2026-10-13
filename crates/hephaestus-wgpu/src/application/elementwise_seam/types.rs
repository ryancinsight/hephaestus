use std::marker::PhantomData;

use crate::infrastructure::device::PipelineCache;

/// Provider-owned implementation of
/// [`ElementwiseOps`](hephaestus_core::ElementwiseOps) over wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuElementwiseOps;

/// Prepared resources for one elementwise dispatch bound to fixed operands.
pub struct PreparedElementwise {
    pub(crate) owner: PipelineCache,
    pub(crate) pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    pub(crate) groups: u32,
    pub(crate) _meta_buffer: Option<crate::infrastructure::pool::PooledBuffer>,
    pub(crate) _scalar_buffer: Option<crate::infrastructure::pool::PooledBuffer>,
}

pub(crate) struct SeamUnaryKernel<Op>(PhantomData<Op>);
pub(crate) struct SeamBinaryKernel<Op>(PhantomData<Op>);
pub(crate) struct SeamScalarKernel<Op>(PhantomData<Op>);
