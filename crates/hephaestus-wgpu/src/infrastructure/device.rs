mod acquisition;
mod allocation;
#[cfg(test)]
mod pool_tests;
mod pools;
mod seam;
#[cfg(test)]
mod tests;
mod transfer;

use std::sync::Arc;
use std::time::Duration;

use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference, HephaestusError, Result};
use std::any::TypeId;

use moirai_sync::ShardedResourcePool;

use crate::infrastructure::pool::{MapCompletionPool, PoolBuffer};

pub(crate) use pools::StagingPoolAccounting;
use pools::{
    STAGING_POOL_MAX_BUFFERS, STAGING_POOL_MAX_BYTES, UNIFORM_POOL_MAX_BUFFERS,
    UNIFORM_POOL_MAX_BYTES,
};

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

// Deadline carried by every device wait this backend performs on a default
// path: readback polls (`download`, `download_owned`, `download_sub_buffer`,
// the decomposition region readbacks), `copy_buffer`, and `synchronize`. An
// unbounded wait is the temporal form of an unbounded queue — a wedged driver
// or a lost device parks the calling thread, which here is a Python binding or
// a solver step, with nothing to act on.
//
// DERIVATION. The bound sits ABOVE the platform's own hang detection, so this
// deadline is the backstop rather than the first reporter:
//   * Windows resets a GPU that misses `TdrDelay` (default 2 s) and allows the
//     driver `TdrDdiDelay` (default 5 s) to unwind before bug-checking, so a
//     genuine hang surfaces through wgpu as a lost device within ~7 s.
//     Source: Microsoft, "Testing and Debugging TDR During Driver Development"
//     (TDR registry keys) — TdrDelay 2 s, TdrDdiDelay 5 s, TdrLimitTime 60 s.
//   * Linux amdgpu's GPU-scheduler `lockup_timeout` defaults to 2000 ms across
//     all queues, inside the same envelope.
// A shorter deadline would fire first and replace an accurate device-lost
// diagnosis with an ambiguous host-side timeout. 30 s is ~4x the 7 s envelope,
// so it fires only where the platform watchdog does not: a compute stack with
// no watchdog, a wedged driver, or work legitimately queued ahead of the
// waited submission for longer than the bound.
//
// UPPER BOUND. 30 s is half of Windows' `TdrLimitTime` (60 s) bug-check window
// and remains a host stall a human notices. Headroom against real work: this
// backend's whole 172-case integration contract suite, every case
// device-resident, completes in ~1.9 s of wall clock, so no wait this backend
// produces is within an order of magnitude of the bound. Callers whose work
// genuinely exceeds it wait explicitly through `download_with_timeout` or
// `WgpuCommandStream::submit_with_timeout` instead of being silently unbounded.
const DEFAULT_DEVICE_WAIT: Duration = Duration::from_secs(30);

#[cfg(test)]
thread_local! {
    /// Test-only deadline override in nanoseconds; 0 selects
    /// [`DEFAULT_DEVICE_WAIT`]. The production bound cannot be reached inside
    /// the configured test budget — doing so needs 30 s of stalled device — so
    /// the override lets a test drive a real submission that is genuinely too
    /// slow for its deadline into the timeout path, rather than asserting the
    /// plumbing against itself. Thread-local, so a case running concurrently
    /// on another thread keeps the production bound.
    static TEST_WAIT_DEADLINE_NS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// The deadline every default device wait in this backend carries.
pub(crate) fn device_wait_deadline() -> Duration {
    #[cfg(test)]
    {
        let override_ns = TEST_WAIT_DEADLINE_NS.with(core::cell::Cell::get);
        if override_ns != 0 {
            return Duration::from_nanos(override_ns);
        }
    }
    DEFAULT_DEVICE_WAIT
}

/// Map a wgpu poll failure onto the typed error a caller can act on.
///
/// An elapsed deadline is not a transfer fault, so it carries its own variant
/// and the bound that expired; every other poll failure stays a transfer
/// failure. Nothing here retries, degrades, or falls back — the caller decides.
pub(crate) fn poll_failure(
    context: &str,
    deadline: Duration,
    error: &wgpu::PollError,
) -> HephaestusError {
    match error {
        wgpu::PollError::Timeout => HephaestusError::DeviceWaitTimeout {
            deadline,
            message: context.to_string(),
        },
        other @ wgpu::PollError::WrongSubmissionIndex(..) => HephaestusError::TransferFailed {
            message: format!("{context}: device poll failed: {other:?}"),
        },
    }
}

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

impl WgpuDevice {
    #[inline]
    const fn wgpu_power_preference(preference: DevicePreference) -> wgpu::PowerPreference {
        match preference {
            DevicePreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            DevicePreference::LowPower => wgpu::PowerPreference::LowPower,
        }
    }

    const fn adapter_preference_rank(
        device_type: wgpu::DeviceType,
        preference: DevicePreference,
    ) -> u8 {
        match (preference, device_type) {
            (DevicePreference::HighPerformance, wgpu::DeviceType::DiscreteGpu)
            | (DevicePreference::LowPower, wgpu::DeviceType::IntegratedGpu) => 0,
            (DevicePreference::HighPerformance, wgpu::DeviceType::IntegratedGpu)
            | (DevicePreference::LowPower, wgpu::DeviceType::DiscreteGpu) => 1,
            (_, wgpu::DeviceType::VirtualGpu) => 2,
            (_, wgpu::DeviceType::Other) => 3,
            (_, wgpu::DeviceType::Cpu) => 4,
        }
    }

    #[inline]
    const fn wgpu_feature(feature: DeviceFeature) -> wgpu::Features {
        match feature {
            DeviceFeature::TimestampQuery => wgpu::Features::TIMESTAMP_QUERY,
            DeviceFeature::ShaderF64 => wgpu::Features::SHADER_F64,
            DeviceFeature::ShaderF16 => wgpu::Features::SHADER_F16,
            DeviceFeature::MappablePrimaryBuffers => wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
            DeviceFeature::ImmediateData => wgpu::Features::IMMEDIATES,
        }
    }

    fn wgpu_features(features: &[DeviceFeature]) -> wgpu::Features {
        features
            .iter()
            .copied()
            .fold(wgpu::Features::empty(), |acc, feature| {
                acc | Self::wgpu_feature(feature)
            })
    }

    #[inline]
    const fn device_limits_from_wgpu(limits: &wgpu::Limits) -> DeviceLimits {
        DeviceLimits {
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_storage_buffers_per_shader_stage: Some(limits.max_storage_buffers_per_shader_stage),
            max_buffers_and_acceleration_structures_per_shader_stage: Some(
                limits.max_buffers_and_acceleration_structures_per_shader_stage,
            ),
            max_immediate_size: limits.max_immediate_size,
        }
    }

    fn wgpu_limits_from_device_limits(required: DeviceLimits) -> wgpu::Limits {
        let downlevel = wgpu::Limits::downlevel_defaults();
        let max_storage_buffers = required
            .max_storage_buffers_per_shader_stage
            .unwrap_or(downlevel.max_storage_buffers_per_shader_stage);
        // Prefer the consumer's explicit combined budget when supplied — only it
        // can account for uniform buffers bound alongside storage buffers. Fall
        // back to deriving from the storage-buffer request for callers that pin
        // only the legacy limit. Storage buffers always count against the
        // aggregate, so it can never fall below the storage request.
        let max_buffers_and_acceleration_structures = required
            .max_buffers_and_acceleration_structures_per_shader_stage
            .unwrap_or(downlevel.max_buffers_and_acceleration_structures_per_shader_stage)
            .max(max_storage_buffers);

        wgpu::Limits {
            max_buffer_size: required.max_buffer_size,
            max_compute_workgroup_size_x: required.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: required.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: required.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: required.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: required.max_compute_workgroup_storage_size,
            max_buffers_and_acceleration_structures_per_shader_stage:
                max_buffers_and_acceleration_structures,
            max_storage_buffers_per_shader_stage: max_storage_buffers,
            max_immediate_size: required.max_immediate_size,
            ..downlevel
        }
    }

    /// WGPU backend default limits mapped into the backend-neutral Hephaestus vocabulary.
    #[must_use]
    pub fn default_device_limits() -> DeviceLimits {
        Self::device_limits_from_wgpu(&wgpu::Limits::default())
    }

    /// WGPU downlevel limits mapped into the backend-neutral Hephaestus vocabulary.
    ///
    /// Use this acquisition baseline when a consumer must retain compatibility
    /// with WGPU downlevel adapters while raising only the limits its kernels
    /// require.
    #[must_use]
    pub fn downlevel_device_limits() -> DeviceLimits {
        Self::device_limits_from_wgpu(&wgpu::Limits::downlevel_defaults())
    }

    /// Wrap an existing device and queue.
    ///
    /// No adapter is available on this path, so no topology snapshot is
    /// reported ([`topology`](Self::topology) returns `None`); the
    /// `try_default*` acquisition paths capture one from the adapter.
    ///
    /// Construction is infallible because the caller already owns a valid
    /// WGPU device and queue pair.
    #[must_use]
    #[inline]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            adapter_info: None,
            adapter_features: None,
            adapter_limits: None,
            topology: None,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            fusion_pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            staging_pool: Arc::new(ShardedResourcePool::new(
                STAGING_POOL_MAX_BUFFERS,
                STAGING_POOL_MAX_BYTES,
            )),
            staging_accounting: Arc::new(StagingPoolAccounting::new()),
            uniform_pool: Arc::new(ShardedResourcePool::new(
                UNIFORM_POOL_MAX_BUFFERS,
                UNIFORM_POOL_MAX_BYTES,
            )),
            // One completion slot per retained staging buffer: retaining more
            // callback state cannot improve the staging pool's warm capacity.
            map_completion_pool: Arc::new(MapCompletionPool::new(STAGING_POOL_MAX_BUFFERS)),
        }
    }

    /// Build a themis topology snapshot from the adapter (atlas ADR 0002:
    /// hephaestus is the provider; themis stays stateless law).
    ///
    /// wgpu deliberately abstracts hardware topology, so only what the API
    /// reports is filled: subgroup (warp/wavefront) width from adapter
    /// limits, and the memory tier inferred from the device type
    /// (integrated GPUs share host DRAM; discrete devices report the
    /// technology-unspecified `Device` tier because wgpu does not expose
    /// HBM-vs-GDDR). Every other capacity is zero per the themis
    /// "unreported fields are zero, never fabricated" contract — the CUDA
    /// backend fills the full set from device attributes.
    fn topology_from_adapter(adapter: &wgpu::Adapter) -> themis::GpuTopology {
        let info = adapter.get_info();
        let memory_tier = match info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => themis::MemoryTier::Dram,
            _ => themis::MemoryTier::Device,
        };
        // WebGPU exposes no SM/register/memory introspection: those
        // capacities are unreported (`None` by type), never fabricated.
        themis::GpuTopology::from_provider(themis::GpuDeviceProperties {
            compute_units: None,
            warp_width: core::num::NonZeroU32::new(info.subgroup_min_size),
            max_threads_per_unit: None,
            registers_per_unit: None,
            shared_mem_per_unit_bytes: None,
            l2_bytes: None,
            memory_tier,
            memory_bytes: None,
        })
    }

    fn device_tier(hint: themis::PlacementHint) -> Result<themis::MemoryTier> {
        let tier = match hint {
            themis::PlacementHint::Tier(tier) => tier,
            _ => themis::MemoryTier::Device,
        };
        if tier == themis::MemoryTier::Device {
            Ok(tier)
        } else {
            Err(HephaestusError::AllocationFailed {
                message: format!(
                    "WGPU cannot guarantee requested memory tier {tier:?}; use Device placement"
                ),
            })
        }
    }

    fn with_adapter_metadata(mut self, adapter: &wgpu::Adapter) -> Self {
        self.topology = Some(Arc::new(Self::topology_from_adapter(adapter)));
        self.adapter_info = Some(adapter.get_info());
        self.adapter_features = Some(adapter.features());
        self.adapter_limits = Some(adapter.limits());
        self
    }

    /// The adapter metadata captured at acquisition, when available.
    ///
    /// `None` when the device was wrapped via [`new`](Self::new) (no adapter
    /// to report from).
    #[must_use]
    #[inline]
    pub fn adapter_info(&self) -> Option<&wgpu::AdapterInfo> {
        self.adapter_info.as_ref()
    }

    /// The adapter features captured at acquisition, when available.
    ///
    /// `None` when the device was wrapped via [`new`](Self::new) (no adapter
    /// to report from).
    #[must_use]
    #[inline]
    pub const fn adapter_features(&self) -> Option<wgpu::Features> {
        self.adapter_features
    }

    /// The adapter limits captured at acquisition, when available.
    ///
    /// `None` when the device was wrapped via [`new`](Self::new) (no adapter
    /// to report from).
    #[must_use]
    #[inline]
    pub fn adapter_limits(&self) -> Option<&wgpu::Limits> {
        self.adapter_limits.as_ref()
    }

    /// The device topology snapshot captured at acquisition, when available.
    ///
    /// `None` when the device was wrapped via [`new`](Self::new) (no adapter
    /// to report from).
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.topology.as_deref()
    }
}

impl WgpuDevice {
    /// Borrow the inner wgpu device for pipeline construction.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the wgpu device `Arc`.
    #[must_use]
    #[inline]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Borrow the wgpu queue `Arc`.
    #[must_use]
    #[inline]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Return the enabled WGPU feature set for this provider.
    #[must_use]
    #[inline]
    pub fn features(&self) -> wgpu::Features {
        self.device.features()
    }

    /// Return the WGPU limits for this provider.
    #[must_use]
    #[inline]
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// The enabled device limits mapped into the backend-neutral Hephaestus vocabulary.
    #[must_use]
    #[inline]
    pub fn device_limits(&self) -> DeviceLimits {
        Self::device_limits_from_wgpu(&self.device.limits())
    }

    /// Return true when the acquired device has `feature` enabled.
    #[must_use]
    #[inline]
    pub fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        self.device.features().contains(Self::wgpu_feature(feature))
    }
}
