//! Acquired device state with shared caches and provider-owned transient pools.

use super::cache::{FusionPipelineCache, PipelineCache};
use super::retention::StagingPoolAccounting;
use crate::infrastructure::pool::{MapCompletionPool, PoolBuffer};
use moirai_sync::ShardedResourcePool;
use std::sync::Arc;

/// An acquired wgpu device + queue pair.
///
/// Clones share the device, caches and transient pools. This is the authoritative
/// adapter/device acquisition for Atlas wgpu consumers; apollo's
/// `apollo-wgpu-helpers` delegates here instead of carrying its own copy.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) adapter_info: Option<wgpu::AdapterInfo>,
    pub(super) adapter_features: Option<wgpu::Features>,
    pub(super) adapter_limits: Option<wgpu::Limits>,
    pub(super) topology: Option<Arc<themis::GpuTopology>>,
    pub(crate) pipeline_cache: PipelineCache,
    pub(crate) fusion_pipeline_cache: FusionPipelineCache,
    pub(super) staging_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    pub(super) uniform_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    pub(super) map_completion_pool: Arc<MapCompletionPool>,
    pub(super) staging_accounting: Arc<StagingPoolAccounting>,
}
