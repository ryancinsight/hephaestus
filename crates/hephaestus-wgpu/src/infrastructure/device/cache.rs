//! Structural keys and ownership for compiled WGPU pipeline caches.

use std::{any::TypeId, sync::Arc};

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
