//! Device construction and adapter metadata.

use super::{
    STAGING_POOL_MAX_BUFFERS, STAGING_POOL_MAX_BYTES, StagingPoolAccounting,
    UNIFORM_POOL_MAX_BUFFERS, UNIFORM_POOL_MAX_BYTES, WgpuDevice,
};
use crate::infrastructure::pool::MapCompletionPool;
use hephaestus_core::{HephaestusError, Result};
use moirai_sync::ShardedResourcePool;
use std::sync::Arc;

impl WgpuDevice {
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

    pub(super) fn device_tier(hint: themis::PlacementHint) -> Result<themis::MemoryTier> {
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

    pub(super) fn with_adapter_metadata(mut self, adapter: &wgpu::Adapter) -> Self {
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
