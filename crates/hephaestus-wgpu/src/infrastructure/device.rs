mod acquisition;
mod allocation;
mod compute;
mod construction;
#[cfg(test)]
mod pool_tests;
mod pools;
#[cfg(test)]
mod tests;
mod transfer;

use std::any::TypeId;
use std::sync::Arc;

use crate::infrastructure::pool::{MapCompletionPool, PoolBuffer};
use moirai_sync::ShardedResourcePool;

pub(crate) use pools::StagingPoolAccounting;
#[cfg(test)]
pub(crate) use transfer::{DEFAULT_DEVICE_WAIT, TEST_WAIT_DEADLINE_NS};
#[cfg(any(test, feature = "decomposition"))]
pub(crate) use transfer::{device_wait_deadline, poll_failure};

/// Pipeline-cache key: kernel-family discriminator, scalar type, block width.
pub(crate) type PipelineKey = (TypeId, TypeId, u32);
/// Collision-safe key for runtime-generated fusion pipelines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FusionPipelineKey {
    pub(crate) family: TypeId,
    pub(crate) scalar: TypeId,
    pub(crate) rank: u32,
    pub(crate) input_count: u32,
    pub(crate) expression: Arc<str>,
}
pub(crate) type PipelineCache = Arc<
    moirai_sync::sync::ConcurrentHashMap<
        PipelineKey,
        Arc<std::sync::OnceLock<wgpu::ComputePipeline>>,
    >,
>;
pub(crate) type FusionPipelineCache = Arc<
    moirai_sync::sync::ConcurrentHashMap<
        FusionPipelineKey,
        Arc<std::sync::OnceLock<std::result::Result<wgpu::ComputePipeline, Arc<str>>>>,
    >,
>;

/// An acquired wgpu device + queue pair.
///
/// Clones share the device, caches and transient pools. This is the authoritative
/// adapter/device acquisition for Atlas wgpu consumers; apollo's
/// `apollo-wgpu-helpers` delegates here instead of carrying its own copy.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: Option<wgpu::AdapterInfo>,
    adapter_features: Option<wgpu::Features>,
    adapter_limits: Option<wgpu::Limits>,
    topology: Option<Arc<themis::GpuTopology>>,
    pub(crate) pipeline_cache: PipelineCache,
    pub(crate) fusion_pipeline_cache: FusionPipelineCache,
    staging_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    uniform_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    map_completion_pool: Arc<MapCompletionPool>,
    staging_accounting: Arc<StagingPoolAccounting>,
}
