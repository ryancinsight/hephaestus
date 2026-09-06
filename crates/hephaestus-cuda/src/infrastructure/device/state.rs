use super::context::CudaContext;
use crate::application::pipeline::{FusionPipelineKey, PipelineKey};
use hephaestus_core::DeviceLimits;
use std::sync::Arc;

/// An acquired CUDA device.
///
/// Owns the selected context and retains its loaded driver through every
/// cloned device, allocation, and compiled module. Acquisition requires a
/// driver at runtime; compiling this crate requires no driver import library.
#[derive(Clone)]
pub struct CudaDevice {
    pub(super) context: Arc<CudaContext>,
    pub(super) limits: DeviceLimits,
    pub(super) features: CudaDeviceFeatures,
    /// Compiled-kernel cache. Slots hold only successful compilations;
    /// failures leave the `OnceLock` empty so the key can retry (see
    /// [`crate::application::pipeline::cached_kernel`]).
    pub(crate) pipeline_cache: Arc<
        moirai_sync::sync::ConcurrentHashMap<
            PipelineKey,
            Arc<std::sync::OnceLock<Arc<crate::infrastructure::compiler::SafeCachedKernel>>>,
        >,
    >,
    /// Runtime-fusion cache keyed by the complete generated expression and
    /// its provider-visible specialization dimensions.
    pub(crate) fusion_pipeline_cache: Arc<
        moirai_sync::sync::ConcurrentHashMap<
            FusionPipelineKey,
            Arc<std::sync::OnceLock<Arc<crate::infrastructure::compiler::SafeCachedKernel>>>,
        >,
    >,
    pub(super) topology: Option<Arc<themis::GpuTopology>>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CudaDeviceFeatures {
    pub(super) shader_f64: bool,
    pub(super) immediate_data: bool,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaDevice").finish_non_exhaustive()
    }
}
