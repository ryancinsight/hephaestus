use core::marker::PhantomData;
use std::alloc::GlobalAlloc;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    validate_buffer_size, validate_slice_alignment, ComputeDevice, ComputeDeviceAcquisition,
    ComputeDeviceCapabilities, DeviceFeature, DeviceLimits, DevicePreference, HephaestusError,
    Result,
};
use std::any::TypeId;
use wgpu::util::DeviceExt;

use crate::infrastructure::buffer::{StagingPointer, WgpuBuffer};
use crate::infrastructure::pool::PoolBuffer;
use moirai_sync::ShardedResourcePool;

use mnemosyne::{MnemosyneAllocator, StandardPolicy, WgpuStagingBackend};

pub(crate) static WGPU_STAGING_ALLOCATOR: MnemosyneAllocator<StandardPolicy, WgpuStagingBackend> =
    MnemosyneAllocator::new();

/// The process's registered staging device: the first `WgpuDevice`
/// constructed. The mnemosyne staging callbacks ([`wgpu_allocate_callback`]/
/// [`wgpu_deallocate_callback`]) are process-global and create mapped buffers
/// on THIS device only; `HostPinned` placement on any other device is
/// rejected with a typed error (see `require_staging_device`) because a
/// mapped buffer belongs to the device that created it. Stored as the `Arc`
/// so identity is checkable via `Arc::ptr_eq` (wgpu resources expose no
/// identity comparison).
pub(crate) static ACTIVE_WGPU_DEVICE: std::sync::OnceLock<Arc<wgpu::Device>> =
    std::sync::OnceLock::new();

std::thread_local! {
    pub(crate) static WGPU_ALLOCATION_USAGE: std::cell::Cell<wgpu::BufferUsages> = const { std::cell::Cell::new(wgpu::BufferUsages::STORAGE) };
}

/// Metadata for an active mapped wgpu buffer tracked by Mnemosyne.
///
/// Internal to the staging-allocator integration (the `wgpu_allocate`/
/// `wgpu_deallocate` callbacks and [`resolve_mapped_buffer`]); not part of the
/// crate's public surface.
pub(crate) struct WgpuMappedBuffer {
    /// The underlying raw wgpu buffer.
    pub(crate) buffer: wgpu::Buffer,
    /// The allocated size in bytes.
    pub(crate) size: usize,
}

/// Thread-safe registry mapping each mapped block's base host address to its
/// underlying `WgpuMappedBuffer` descriptor.
///
/// A `BTreeMap` (keyed by base address) is used rather than a `HashMap` so that
/// resolving a sub-allocated pointer to its containing block is an `O(log n)`
/// range query ([`resolve_mapped_buffer`]) instead of an `O(n)` linear scan
/// while holding only a shared read lock.
pub(crate) static WGPU_MAPPED_BUFFERS: std::sync::LazyLock<
    std::sync::RwLock<std::collections::BTreeMap<usize, WgpuMappedBuffer>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::BTreeMap::new()));

/// Resolves the `wgpu::Buffer` whose mapped host range contains `ptr`.
///
/// The Mnemosyne staging allocator may return a pointer offset into a larger
/// mapped block, so the registry is queried for the greatest base address
/// `<= ptr` and the result is range-checked for containment. The `BTreeMap`
/// range query keeps the shared-read critical section `O(log n)`; the
/// returned `wgpu::Buffer` is a cheap `Arc` handle clone.
fn resolve_mapped_buffer(ptr: *mut u8) -> Result<wgpu::Buffer> {
    let block_addr = ptr as usize;
    let mapped = WGPU_MAPPED_BUFFERS.read().unwrap();
    if let Some((&base_addr, mapped_buf)) = mapped.range(..=block_addr).next_back() {
        if block_addr < base_addr + mapped_buf.size {
            return Ok(mapped_buf.buffer.clone());
        }
    }
    Err(HephaestusError::AllocationFailed {
        message: format!("Buffer not found in WGPU_MAPPED_BUFFERS registry for ptr {ptr:p}"),
    })
}

/// Mnemosyne staging-backend allocation callback.
///
/// # Pointer-validity invariant (mapped-range escape)
///
/// The returned pointer comes from `get_mapped_range{_mut}` on a buffer that
/// STAYS MAPPED for the allocation's whole lifetime: it is unmapped only in
/// [`wgpu_deallocate_callback`] after removal from the registry. wgpu pins the
/// host allocation of a mapped buffer for as long as the mapping is active, so
/// the pointer outliving the temporary `BufferView{Mut}` guard is sound under
/// that documented mapping lifetime, not under the guard's lifetime. Unmapping
/// while a Mnemosyne sub-allocation is live would invalidate this; the
/// registry's remove-then-unmap ordering prevents it.
unsafe extern "C" fn wgpu_allocate_callback(size: usize) -> *mut u8 {
    let device = match ACTIVE_WGPU_DEVICE.get() {
        Some(d) => d,
        None => return core::ptr::null_mut(),
    };
    let usage = WGPU_ALLOCATION_USAGE.with(|u| u.get());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hephaestus-mnemosyne-staging"),
        size: size as u64,
        usage,
        mapped_at_creation: false,
    });

    let slice = buffer.slice(..);
    let ptr = if usage.contains(wgpu::BufferUsages::MAP_WRITE) {
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Write, move |res| {
            let _ = tx.send(res);
        });
        if device.poll(wgpu::PollType::Wait).is_err() {
            return core::ptr::null_mut();
        }
        if rx.recv().is_ok_and(|res| res.is_ok()) {
            slice.get_mapped_range_mut().as_mut_ptr()
        } else {
            return core::ptr::null_mut();
        }
    } else if usage.contains(wgpu::BufferUsages::MAP_READ) {
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        if device.poll(wgpu::PollType::Wait).is_err() {
            return core::ptr::null_mut();
        }
        if rx.recv().is_ok_and(|res| res.is_ok()) {
            slice.get_mapped_range().as_ptr() as *mut u8
        } else {
            return core::ptr::null_mut();
        }
    } else {
        return core::ptr::null_mut();
    };

    if ptr.is_null() {
        return core::ptr::null_mut();
    }

    let mut mapped = WGPU_MAPPED_BUFFERS.write().unwrap();
    mapped.insert(ptr as usize, WgpuMappedBuffer { buffer, size });

    ptr
}

unsafe extern "C" fn wgpu_deallocate_callback(ptr: *mut u8, _size: usize) -> bool {
    let mut mapped = WGPU_MAPPED_BUFFERS.write().unwrap();
    if let Some(mapped_buf) = mapped.remove(&(ptr as usize)) {
        mapped_buf.buffer.unmap();
        true
    } else {
        false
    }
}

/// Pipeline-cache key: kernel-family discriminator, scalar type, block width.
pub(crate) type PipelineKey = (TypeId, TypeId, u32);
pub(crate) type PipelineCache = Arc<
    moirai_sync::sync::ConcurrentHashMap<
        PipelineKey,
        Arc<std::sync::OnceLock<wgpu::ComputePipeline>>,
    >,
>;

// Pool budgets. `ShardedResourcePool` divides both caps by its 4 thread-
// affine shards and recycles to the CALLER's shard only, so the effective
// single-threaded retention is `max_buffers / 4` buffers and an item larger
// than `max_bytes / 4` is never pooled. Budgets below are chosen against that
// division, not the nominal totals.
//
// Staging: 8 buffers (2/shard — transfers use one at a time) with a 512 MiB
// byte budget so a single staging buffer up to 128 MiB (/4) still pools;
// 16 MiB (the previous /4 ceiling) is smaller than routine volumetric
// readbacks (e.g. a 256³ f32 volume is 64 MiB), which made every large
// download allocate-and-destroy. The budget is a retention CEILING, not a
// preallocation: nothing is retained unless a transfer of that size happened,
// and `clear_transient_pools` releases retained buffers on demand.
const STAGING_POOL_MAX_BUFFERS: usize = 8;
const STAGING_POOL_MAX_BYTES: u64 = 512 * 1024 * 1024;
// Uniforms: metadata blocks of ≲256 B; ops acquire up to three per call
// (`matmul_into`, `kron_into`), so 2/shard (the old 8/4) forced an
// allocate-evict cycle on every 3-uniform call from one thread. 32 (8/shard)
// retains at most ~8 KiB of uniforms per shard against the 1 MiB/4 budget.
const UNIFORM_POOL_MAX_BUFFERS: usize = 32;
const UNIFORM_POOL_MAX_BYTES: u64 = 1024 * 1024;

/// An acquired wgpu device + queue pair.
///
/// `Clone` is cheap (three `Arc` clones). This is the single authoritative
/// adapter/device acquisition for Atlas wgpu consumers; apollo's
/// `apollo-wgpu-helpers` delegates here instead of carrying its own copy.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: Option<wgpu::AdapterInfo>,
    adapter_limits: Option<wgpu::Limits>,
    topology: Option<Arc<themis::GpuTopology>>,
    pub(crate) pipeline_cache: PipelineCache,
    pub(crate) staging_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    pub(crate) uniform_pool: Arc<ShardedResourcePool<PoolBuffer>>,
}

impl WgpuDevice {
    #[inline]
    const fn wgpu_power_preference(preference: DevicePreference) -> wgpu::PowerPreference {
        match preference {
            DevicePreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            DevicePreference::LowPower => wgpu::PowerPreference::LowPower,
        }
    }

    #[inline]
    const fn wgpu_feature(feature: DeviceFeature) -> wgpu::Features {
        match feature {
            DeviceFeature::TimestampQuery => wgpu::Features::TIMESTAMP_QUERY,
            DeviceFeature::ShaderF64 => wgpu::Features::SHADER_F64,
            DeviceFeature::ShaderF16 => wgpu::Features::SHADER_F16,
            DeviceFeature::MappablePrimaryBuffers => wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
            DeviceFeature::PushConstants => wgpu::Features::PUSH_CONSTANTS,
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
            max_push_constant_size: limits.max_push_constant_size,
        }
    }

    fn wgpu_limits_from_device_limits(required: DeviceLimits) -> wgpu::Limits {
        wgpu::Limits {
            max_buffer_size: required.max_buffer_size,
            max_compute_workgroup_size_x: required.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: required.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: required.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: required.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: required.max_compute_workgroup_storage_size,
            max_storage_buffers_per_shader_stage: required
                .max_storage_buffers_per_shader_stage
                .unwrap_or_else(|| wgpu::Limits::default().max_storage_buffers_per_shader_stage),
            max_push_constant_size: required.max_push_constant_size,
            ..wgpu::Limits::default()
        }
    }

    /// WGPU backend default limits mapped into the backend-neutral Hephaestus vocabulary.
    #[must_use]
    pub fn default_device_limits() -> DeviceLimits {
        Self::device_limits_from_wgpu(&wgpu::Limits::default())
    }

    /// Wrap an existing device and queue.
    ///
    /// No adapter is available on this path, so no topology snapshot is
    /// reported ([`topology`](Self::topology) returns `None`); the
    /// `try_default*` acquisition paths capture one from the adapter.
    #[must_use]
    #[inline]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        // First-wins registration: the first constructed device becomes the
        // process staging device; later devices still compute normally but
        // `HostPinned` placement on them is rejected (require_staging_device).
        let _ = ACTIVE_WGPU_DEVICE.set(device.clone());

        mnemosyne_backend::WGPU_ALLOCATE_CALLBACK.store(
            wgpu_allocate_callback as *mut core::ffi::c_void,
            std::sync::atomic::Ordering::Release,
        );
        mnemosyne_backend::WGPU_DEALLOCATE_CALLBACK.store(
            wgpu_deallocate_callback as *mut core::ffi::c_void,
            std::sync::atomic::Ordering::Release,
        );

        Self {
            device,
            queue,
            adapter_info: None,
            adapter_limits: None,
            topology: None,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            staging_pool: Arc::new(ShardedResourcePool::new(
                STAGING_POOL_MAX_BUFFERS,
                STAGING_POOL_MAX_BYTES,
            )),
            uniform_pool: Arc::new(ShardedResourcePool::new(
                UNIFORM_POOL_MAX_BUFFERS,
                UNIFORM_POOL_MAX_BYTES,
            )),
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
        let limits = adapter.limits();
        let info = adapter.get_info();
        let memory_tier = match info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => themis::MemoryTier::Dram,
            _ => themis::MemoryTier::Device,
        };
        themis::GpuTopology::from_provider(themis::GpuDeviceProperties {
            compute_units: 0,
            warp_width: limits.min_subgroup_size,
            max_threads_per_unit: 0,
            registers_per_unit: 0,
            shared_mem_per_unit_bytes: 0,
            l2_bytes: 0,
            memory_tier,
            memory_bytes: 0,
        })
    }

    /// Require that `self` is the process's registered staging device before
    /// a `HostPinned` allocation.
    ///
    /// The mnemosyne staging callbacks are process-global and bound to the
    /// FIRST constructed device; a mapped buffer they create belongs to that
    /// `wgpu::Device`, and binding or copying it on another device is a wgpu
    /// validation error surfaced far from the cause. Rejecting here names the
    /// real constraint at the allocation site instead.
    ///
    /// Identity is `Arc` pointer identity: clones of the registered
    /// `WgpuDevice` share the `Arc` and pass; a device wrapped separately via
    /// [`new`](Self::new) around the same `wgpu::Device` is conservatively
    /// rejected (wgpu resources expose no identity comparison).
    fn require_staging_device(&self) -> Result<()> {
        match ACTIVE_WGPU_DEVICE.get() {
            Some(registered) if Arc::ptr_eq(registered, &self.device) => Ok(()),
            Some(_) => Err(HephaestusError::AllocationFailed {
                message: "HostPinned placement is only available on the process's registered \
                          staging device (the first WgpuDevice constructed); allocate HostPinned \
                          there or use the default Device tier on this device"
                    .to_string(),
            }),
            None => Err(HephaestusError::AllocationFailed {
                message: "no registered staging device for HostPinned placement".to_string(),
            }),
        }
    }

    fn with_adapter_metadata(mut self, adapter: &wgpu::Adapter) -> Self {
        self.topology = Some(Arc::new(Self::topology_from_adapter(adapter)));
        self.adapter_info = Some(adapter.get_info());
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

    /// Acquire a default high-performance adapter and device.
    ///
    /// `label` becomes the wgpu device label. Uses
    /// [`wgpu::Limits::downlevel_defaults`]; for custom limits use
    /// [`try_default_with_limits`](Self::try_default_with_limits).
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    #[inline]
    pub fn try_default(label: &str) -> Result<Self> {
        Self::try_default_with_limits(label, wgpu::Limits::downlevel_defaults())
    }

    /// Acquire a default adapter and device with custom limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    #[inline]
    pub fn try_default_with_limits(label: &str, required_limits: wgpu::Limits) -> Result<Self> {
        Self::try_default_with_features_and_limits(label, wgpu::Features::empty(), required_limits)
    }

    /// Acquire a default adapter and device with custom features and limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_default_with_features_and_limits(
        label: &str,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            required_limits,
            wgpu::PowerPreference::HighPerformance,
            |_| required_features,
        )
    }

    /// Acquire a default adapter and device, enabling optional features only
    /// when the selected adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_default_with_optional_features_and_limits(
        label: &str,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            required_limits,
            wgpu::PowerPreference::HighPerformance,
            |adapter| adapter.features() & optional_features,
        )
    }

    /// Acquire an adapter matching `device_preference`, enabling optional
    /// features only when the selected adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_optional_features_and_limits(
        label: &str,
        device_preference: DevicePreference,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_with_power_preference_and_optional_features_and_limits(
            label,
            Self::wgpu_power_preference(device_preference),
            optional_features,
            required_limits,
        )
    }

    /// Acquire an adapter matching `device_preference`, enabling optional
    /// Hephaestus features only when the selected adapter reports support for
    /// them. Uses the backend's default WGPU limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_optional_device_features(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
    ) -> Result<Self> {
        Self::try_with_device_preference_and_optional_features_and_limits(
            label,
            device_preference,
            Self::wgpu_features(optional_features),
            wgpu::Limits::default(),
        )
    }

    /// Acquire an adapter matching `device_preference`, enabling optional
    /// Hephaestus features when supported and applying backend-neutral required
    /// compute limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_optional_device_features_and_limits(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        Self::try_with_device_preference_and_optional_features_and_limits(
            label,
            device_preference,
            Self::wgpu_features(optional_features),
            Self::wgpu_limits_from_device_limits(required_limits),
        )
    }

    /// Acquire an adapter matching `power_preference`, enabling optional
    /// features only when the selected adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_power_preference_and_optional_features_and_limits(
        label: &str,
        power_preference: wgpu::PowerPreference,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            required_limits,
            power_preference,
            |adapter| adapter.features() & optional_features,
        )
    }

    /// Acquire an adapter matching `device_preference`, deriving both required
    /// features and required limits from the selected adapter.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_adapter_config(
        label: &str,
        device_preference: DevicePreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Self> {
        Self::try_with_power_preference_and_adapter_config(
            label,
            Self::wgpu_power_preference(device_preference),
            select_features,
            select_limits,
        )
    }

    /// Acquire an adapter matching `power_preference`, deriving both required
    /// features and required limits from the selected adapter.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_power_preference_and_adapter_config(
        label: &str,
        power_preference: wgpu::PowerPreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_config(
            label,
            power_preference,
            select_features,
            select_limits,
        )
    }

    /// Enumerate adapters and create devices for those accepted by
    /// `accept_adapter`, deriving each device descriptor from the selected
    /// adapter.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for any accepted adapter.
    pub fn try_enumerate_with_adapter_config(
        label_prefix: &str,
        max_devices: usize,
        accept_adapter: impl Fn(&wgpu::AdapterInfo) -> bool,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Vec<Self>> {
        let mut desc = wgpu::InstanceDescriptor::from_env_or_default();
        desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(&desc);
        let mut devices = Vec::new();

        for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            let info = adapter.get_info();
            if !accept_adapter(&info) {
                continue;
            }

            let label = format!("{label_prefix}: {}", info.name);
            let required_features = select_features(&adapter);
            let required_limits = select_limits(&adapter);
            devices.push(Self::try_from_adapter_with_features_and_limits(
                &label,
                &adapter,
                required_features,
                required_limits,
            )?);

            if devices.len() >= max_devices {
                break;
            }
        }

        Ok(devices)
    }

    /// Create a device from a caller-selected adapter, enabling optional
    /// features only when that adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for the supplied adapter.
    pub fn try_from_adapter_with_optional_features_and_limits(
        label: &str,
        adapter: &wgpu::Adapter,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        let required_features = adapter.features() & optional_features;
        Self::try_from_adapter_with_features_and_limits(
            label,
            adapter,
            required_features,
            required_limits,
        )
    }

    /// Create a device from a caller-selected adapter with exact required
    /// features.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for the supplied adapter.
    pub fn try_from_adapter_with_features_and_limits(
        label: &str,
        adapter: &wgpu::Adapter,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        let (device, queue) = moirai::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features,
            required_limits,
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| HephaestusError::DeviceUnavailable {
            message: error.to_string(),
        })?;
        Ok(Self::new(Arc::new(device), Arc::new(queue)).with_adapter_metadata(adapter))
    }

    fn try_default_with_adapter_features_and_limits(
        label: &str,
        required_limits: wgpu::Limits,
        power_preference: wgpu::PowerPreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
    ) -> Result<Self> {
        Self::try_default_with_adapter_config(label, power_preference, select_features, move |_| {
            required_limits.clone()
        })
    }

    fn try_default_with_adapter_config(
        label: &str,
        power_preference: wgpu::PowerPreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Self> {
        let try_acquire = |instance: &wgpu::Instance| -> Option<Self> {
            let try_device = |adapter: &wgpu::Adapter| -> std::result::Result<
                (wgpu::Device, wgpu::Queue),
                wgpu::RequestDeviceError,
            > {
                let required_features = select_features(adapter);
                let required_limits = select_limits(adapter);
                moirai::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some(label),
                    required_features,
                    required_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                }))
            };

            // Try High Performance hardware adapter first
            if let Ok(adapter) =
                moirai::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
            {
                if let Ok((device, queue)) = try_device(&adapter) {
                    return Some(
                        Self::new(Arc::new(device), Arc::new(queue))
                            .with_adapter_metadata(&adapter),
                    );
                }
            }

            // Fallback to software/fallback adapter
            if let Ok(fallback_adapter) =
                moirai::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                }))
            {
                if let Ok((device, queue)) = try_device(&fallback_adapter) {
                    return Some(
                        Self::new(Arc::new(device), Arc::new(queue))
                            .with_adapter_metadata(&fallback_adapter),
                    );
                }
            }
            None
        };

        let has_env =
            std::env::var("WGPU_BACKENDS").is_ok() || std::env::var("WGPU_BACKEND").is_ok();

        if cfg!(target_os = "windows") && !has_env {
            // Try DX12 first to completely avoid Vulkan driver access violations in parallel nextest runs.
            let mut desc = wgpu::InstanceDescriptor::from_env_or_default();
            desc.backends = wgpu::Backends::DX12;
            let instance = wgpu::Instance::new(&desc);
            if let Some(device) = try_acquire(&instance) {
                return Ok(device);
            }

            // Fallback to Vulkan if DX12 is unavailable on the host
            let mut desc = wgpu::InstanceDescriptor::from_env_or_default();
            desc.backends = wgpu::Backends::VULKAN;
            let instance = wgpu::Instance::new(&desc);
            if let Some(device) = try_acquire(&instance) {
                return Ok(device);
            }
        } else {
            let desc = wgpu::InstanceDescriptor::from_env_or_default();
            let instance = wgpu::Instance::new(&desc);
            if let Some(device) = try_acquire(&instance) {
                return Ok(device);
            }
        }

        Err(HephaestusError::AdapterUnavailable {
            message: "No compatible GPU adapter or device could be acquired.".to_string(),
        })
    }

    /// Acquire a default adapter and device, specifically targeting the Metal backend.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no Metal adapter can be acquired.
    pub fn try_metal(label: &str) -> Result<Self> {
        let try_acquire = |instance: &wgpu::Instance| -> Option<Self> {
            let try_device = |adapter: &wgpu::Adapter| -> std::result::Result<
                (wgpu::Device, wgpu::Queue),
                wgpu::RequestDeviceError,
            > {
                moirai::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some(label),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                }))
            };

            if let Ok(adapter) =
                moirai::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
            {
                let topology = Self::topology_from_adapter(&adapter);
                if let Ok((device, queue)) = try_device(&adapter) {
                    let mut acquired = Self::new(Arc::new(device), Arc::new(queue));
                    acquired.topology = Some(Arc::new(topology));
                    return Some(acquired);
                }
            }
            None
        };

        let mut desc = wgpu::InstanceDescriptor::from_env_or_default();
        desc.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(&desc);
        if let Some(device) = try_acquire(&instance) {
            Ok(device)
        } else {
            Err(HephaestusError::AdapterUnavailable {
                message: "No compatible Metal GPU adapter or device could be acquired.".to_string(),
            })
        }
    }

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

    /// Exact byte size of `len` elements of `T`.
    pub(crate) fn byte_size<T>(len: usize) -> Result<u64> {
        len.checked_mul(core::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!(
                    "buffer length {len} overflows byte size for {}-byte elements",
                    core::mem::size_of::<T>()
                ),
            })
    }

    /// Size in bytes of `len` elements of `T`, padded to wgpu copy alignment.
    fn padded_size<T>(len: usize) -> Result<u64> {
        let bytes = Self::byte_size::<T>(len)?;
        Self::aligned_size(bytes, wgpu::COPY_BUFFER_ALIGNMENT)
    }

    /// Align `size` upward to `alignment`.
    fn aligned_size(size: u64, alignment: u64) -> Result<u64> {
        size.checked_add(alignment - 1)
            .map(|bytes| (bytes / alignment) * alignment)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!("buffer byte size {size} cannot be aligned to {alignment} bytes"),
            })
    }

    /// Retrieve a staging buffer of size >= size from the bounded pool, or
    /// create a new one. The size is automatically aligned to
    /// `wgpu::MAP_ALIGNMENT` (8 bytes).
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AllocationFailed`] when `size` cannot be aligned
    /// without overflowing `u64`.
    pub fn get_staging_buffer(&self, size: u64) -> Result<wgpu::Buffer> {
        let staging_size = Self::aligned_size(size, wgpu::MAP_ALIGNMENT)?;
        Ok(
            if let Some(buffer) = self.staging_pool.take_at_least(staging_size) {
                buffer.0
            } else {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("hephaestus-recycled-staging"),
                    size: staging_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            },
        )
    }

    /// Return a staging buffer back to the bounded pool for reuse.
    pub fn recycle_staging_buffer(&self, buffer: wgpu::Buffer) {
        self.staging_pool.recycle(PoolBuffer(buffer));
    }

    /// Retrieve a uniform buffer of size ≥ `size` from the pool, or create
    /// one. Retention is bounded by count and bytes. Contents are written
    /// with `queue.write_buffer`, which is ordered on the queue timeline
    /// relative to submissions, so a recycled uniform can be rewritten for
    /// the next dispatch without racing in-flight work on the same queue.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AllocationFailed`] when `size` cannot be aligned
    /// without overflowing `u64`.
    pub fn get_uniform_buffer(&self, size: u64) -> Result<wgpu::Buffer> {
        let uniform_size = Self::aligned_size(size, wgpu::COPY_BUFFER_ALIGNMENT)?;
        Ok(
            if let Some(buffer) = self.uniform_pool.take_at_least(uniform_size) {
                buffer.0
            } else {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("hephaestus-recycled-uniform"),
                    size: uniform_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            },
        )
    }

    /// Return a uniform buffer back to the bounded pool for reuse.
    pub fn recycle_uniform_buffer(&self, buffer: wgpu::Buffer) {
        self.uniform_pool.recycle(PoolBuffer(buffer));
    }

    /// Drop transient buffers retained for reuse.
    ///
    /// The bounded pools avoid repeated staging and uniform allocations on hot
    /// paths. Bindings and short-lived host integrations can call this at an
    /// ownership boundary to release cached allocations before the host runtime
    /// tears down GPU state.
    #[inline]
    pub fn clear_transient_pools(&self) {
        self.staging_pool.clear();
        self.uniform_pool.clear();
    }

    /// Copy a subset of a device buffer's contents into a host slice (device→host).
    ///
    /// The transfer starts at element `offset` in the device buffer and copies
    /// `out.len()` elements. The range `offset..offset + out.len()` must be within
    /// `buffer.len`.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::LengthMismatch`] if the requested range falls outside the buffer bounds.
    /// [`HephaestusError::AllocationFailed`] if element byte conversion overflows `u64`.
    pub fn download_sub_buffer<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        offset: usize,
        out: &mut [T],
    ) -> Result<()> {
        validate_slice_alignment(out)?;
        let end =
            offset
                .checked_add(out.len())
                .ok_or_else(|| HephaestusError::AllocationFailed {
                    message: format!("offset {offset} + out.len() {} overflows usize", out.len()),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        if out.is_empty() {
            return Ok(());
        }

        // byte_size::<T>(offset) = offset * size_of::<T>() with checked overflow → u64.
        let byte_offset = Self::byte_size::<T>(offset)?;
        let byte_len = Self::byte_size::<T>(out.len())?;
        let padded = Self::padded_size::<T>(out.len())?;
        self.stage_and_read(
            &buffer.buffer,
            byte_offset,
            padded,
            byte_len,
            out,
            "hephaestus-download-sub",
        )
    }

    /// Overwrite a subset of a device buffer with host data (host→device).
    ///
    /// Writes `host.len()` elements starting at element `offset` in the device buffer.
    /// The range `offset..offset + host.len()` must be within `buffer.len`.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::LengthMismatch`] if the requested range falls outside the buffer bounds.
    /// [`HephaestusError::AllocationFailed`] if element byte conversion overflows `u64`.
    pub fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        validate_slice_alignment(host)?;
        let end =
            offset
                .checked_add(host.len())
                .ok_or_else(|| HephaestusError::AllocationFailed {
                    message: format!(
                        "offset {offset} + host.len() {} overflows usize",
                        host.len()
                    ),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        if host.is_empty() {
            return Ok(());
        }

        // byte_size reuses the existing checked multiplication (offset * size_of::<T>()).
        let byte_offset = Self::byte_size::<T>(offset)?;
        self.queue
            .write_buffer(buffer.raw(), byte_offset, bytemuck::cast_slice(host));
        Ok(())
    }
}

impl ComputeDevice for WgpuDevice {
    type Buffer<T: Pod> = WgpuBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "wgpu"
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<WgpuBuffer<T>> {
        validate_buffer_size::<T>(len)?;
        let tier = match hint {
            themis::PlacementHint::Tier(t) => t,
            _ => themis::MemoryTier::Device,
        };
        if tier == themis::MemoryTier::HostPinned {
            self.require_staging_device()?;
            let padded_len = Self::padded_size::<T>(len)?;
            let padded_len_usize =
                usize::try_from(padded_len).map_err(|_| HephaestusError::AllocationFailed {
                    message: "padded length overflows usize".to_string(),
                })?;
            WGPU_ALLOCATION_USAGE.with(|u| {
                u.set(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            });
            let layout = std::alloc::Layout::from_size_align(padded_len_usize, 8).map_err(|e| {
                HephaestusError::AllocationFailed {
                    message: format!("invalid layout: {e}"),
                }
            })?;
            // SAFETY: `layout` has non-zero-checked size and valid alignment
            // (constructed just above); the allocator routes to
            // `wgpu_allocate_callback`, whose returned pointer is valid for
            // `layout.size()` bytes until deallocated (mapped-range invariant
            // documented on the callback).
            let ptr = unsafe { WGPU_STAGING_ALLOCATOR.alloc(layout) };
            if ptr.is_null() {
                return Err(HephaestusError::AllocationFailed {
                    message: "Mnemosyne WgpuStagingBackend allocation returned null".to_string(),
                });
            }
            let buffer = resolve_mapped_buffer(ptr)?;
            let staging_ptr = Arc::new(StagingPointer {
                ptr,
                size: padded_len_usize,
            });
            Ok(WgpuBuffer {
                buffer,
                len,
                tier,
                staging_ptr: Some(staging_ptr),
                marker: PhantomData,
            })
        } else {
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            // WebGPU guarantees newly created buffers are zero-initialized.
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hephaestus-storage"),
                size: Self::padded_size::<T>(len)?,
                usage,
                mapped_at_creation: false,
            });
            Ok(WgpuBuffer {
                buffer,
                len,
                tier,
                staging_ptr: None,
                marker: PhantomData,
            })
        }
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<WgpuBuffer<T>> {
        validate_slice_alignment(host)?;
        let byte_len = Self::byte_size::<T>(host.len())?;
        let padded_len = Self::padded_size::<T>(host.len())?;
        let tier = match hint {
            themis::PlacementHint::Tier(t) => t,
            _ => themis::MemoryTier::Device,
        };
        if tier == themis::MemoryTier::HostPinned {
            self.require_staging_device()?;
            let padded_len_usize =
                usize::try_from(padded_len).map_err(|_| HephaestusError::AllocationFailed {
                    message: "padded length overflows usize".to_string(),
                })?;
            WGPU_ALLOCATION_USAGE.with(|u| {
                u.set(wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC);
            });
            let layout = std::alloc::Layout::from_size_align(padded_len_usize, 8).map_err(|e| {
                HephaestusError::AllocationFailed {
                    message: format!("invalid layout: {e}"),
                }
            })?;
            // SAFETY: `layout` has non-zero-checked size and valid alignment
            // (constructed just above); the allocator routes to
            // `wgpu_allocate_callback`, whose returned pointer is valid for
            // `layout.size()` bytes until deallocated (mapped-range invariant
            // documented on the callback).
            let ptr = unsafe { WGPU_STAGING_ALLOCATOR.alloc(layout) };
            if ptr.is_null() {
                return Err(HephaestusError::AllocationFailed {
                    message: "Mnemosyne WgpuStagingBackend allocation returned null".to_string(),
                });
            }
            let byte_len_usize = usize::try_from(byte_len).expect("invariant: byte_len fits usize");
            // SAFETY: `ptr` is valid for `padded_len_usize >= byte_len_usize`
            // writes (allocated above, null-checked); the source is
            // `byte_len_usize` readable bytes of `host` (`T: Pod`); the two
            // ranges cannot overlap — one is a fresh mapped-GPU-buffer range,
            // the other caller host memory.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytemuck::cast_slice(host).as_ptr(),
                    ptr,
                    byte_len_usize,
                );
            }
            let buffer = resolve_mapped_buffer(ptr)?;
            let staging_ptr = Arc::new(StagingPointer {
                ptr,
                size: padded_len_usize,
            });
            Ok(WgpuBuffer {
                buffer,
                len: host.len(),
                tier,
                staging_ptr: Some(staging_ptr),
                marker: PhantomData,
            })
        } else {
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let buffer = if padded_len == 0 {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("hephaestus-upload"),
                    size: 0,
                    usage,
                    mapped_at_creation: false,
                })
            } else {
                // Both byte_len and padded_len come from byte_size/padded_size which
                // already ensure the value fits usize (they start from usize*size_of<T>).
                let byte_len_usize = usize::try_from(byte_len).expect(
                    "invariant: byte_len <= usize::MAX (derived from host.len() * size_of::<T>())",
                );
                let padded_len_usize = usize::try_from(padded_len)
                    .expect("invariant: padded_len <= usize::MAX (padded_size rounds byte_len up by at most alignment-1)");
                let mut padded = vec![0u8; padded_len_usize];
                padded[..byte_len_usize].copy_from_slice(bytemuck::cast_slice(host));
                (*self.device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hephaestus-upload"),
                    contents: &padded,
                    usage,
                })
            };
            Ok(WgpuBuffer {
                buffer,
                len: host.len(),
                tier,
                staging_ptr: None,
                marker: PhantomData,
            })
        }
    }

    fn download<T: Pod>(&self, buffer: &WgpuBuffer<T>, out: &mut [T]) -> Result<()> {
        validate_slice_alignment(out)?;
        if out.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: out.len(),
                device_len: buffer.len,
            });
        }
        if out.is_empty() {
            return Ok(());
        }

        let byte_len = Self::byte_size::<T>(buffer.len)?;
        let padded = Self::padded_size::<T>(buffer.len)?;
        self.stage_and_read(
            &buffer.buffer,
            0,
            padded,
            byte_len,
            out,
            "hephaestus-download",
        )
    }

    fn write_buffer<T: Pod>(&self, buffer: &WgpuBuffer<T>, host: &[T]) -> Result<()> {
        validate_slice_alignment(host)?;
        if host.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: buffer.len,
            });
        }
        self.queue
            .write_buffer(buffer.raw(), 0, bytemuck::cast_slice(host));
        Ok(())
    }

    #[inline]
    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        WgpuDevice::write_sub_buffer(self, buffer, offset, host)
    }

    fn synchronize(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| HephaestusError::TransferFailed {
                message: format!("device poll failed: {e:?}"),
            })?;
        Ok(())
    }
}

impl ComputeDeviceCapabilities for WgpuDevice {
    #[inline]
    fn device_limits(&self) -> DeviceLimits {
        WgpuDevice::device_limits(self)
    }

    #[inline]
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        WgpuDevice::supports_device_feature(self, feature)
    }
}

impl ComputeDeviceAcquisition for WgpuDevice {
    fn try_acquire_device(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        Self::try_with_device_preference_and_optional_device_features_and_limits(
            label,
            device_preference,
            optional_features,
            required_limits,
        )
    }

    fn try_acquire_devices(
        label_prefix: &str,
        max_devices: usize,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Vec<Self>> {
        Self::try_enumerate_with_adapter_config(
            label_prefix,
            max_devices,
            |info| !matches!(info.backend, wgpu::Backend::BrowserWebGpu),
            |adapter| adapter.features() & Self::wgpu_features(optional_features),
            |_| Self::wgpu_limits_from_device_limits(required_limits),
        )
    }
}

impl WgpuDevice {
    /// Core GPU→host transfer: copy `padded` bytes from `src_buf[byte_offset..]`
    /// into a staging buffer, map it, and write exactly `byte_len` bytes into `out`.
    ///
    /// `byte_len` ≤ `padded` must hold; `padded` must fit the alignment required by
    /// `wgpu::COPY_BUFFER_ALIGNMENT`. This is the SSOT for all synchronous
    /// device→host readback paths.
    fn stage_and_read<T: Pod>(
        &self,
        src_buf: &wgpu::Buffer,
        byte_offset: u64,
        padded: u64,
        byte_len: u64,
        out: &mut [T],
        label: &str,
    ) -> Result<()> {
        let raw_staging = self.get_staging_buffer(padded)?;
        let staging_size = raw_staging.size();
        let staging = crate::infrastructure::pool::staging_guard(self.clone(), raw_staging);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_buffer_to_buffer(src_buf, byte_offset, &staging, 0, padded);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..staging_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // send() only fails if the receiver was dropped; the receiver is alive
            // in the same synchronous frame until after poll() returns below.
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| HephaestusError::TransferFailed {
                message: format!("device poll failed: {e:?}"),
            })?;
        receiver
            .recv()
            .map_err(|_| HephaestusError::TransferFailed {
                message: "map_async callback dropped".to_string(),
            })?
            .map_err(|e| HephaestusError::TransferFailed {
                message: format!("buffer mapping failed: {e:?}"),
            })?;

        // byte_len comes from byte_size::<T>(n) = n * size_of::<T>(), which fits usize.
        let byte_len_usize = usize::try_from(byte_len)
            .expect("invariant: byte_len fits usize (derived from element count * size_of::<T>())");
        let mapped = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&mapped[..byte_len_usize]));
        drop(mapped);
        staging.unmap();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_size_aligns_to_copy_boundary() {
        match WgpuDevice::byte_size::<u32>(3) {
            Ok(bytes) => assert_eq!(bytes, 12),
            Err(error) => panic!("expected exact byte size, got {error:?}"),
        }
        match WgpuDevice::padded_size::<u8>(3) {
            Ok(bytes) => assert_eq!(bytes, wgpu::COPY_BUFFER_ALIGNMENT),
            Err(error) => panic!("expected padded byte size, got {error:?}"),
        }
        match WgpuDevice::padded_size::<u32>(0) {
            Ok(bytes) => assert_eq!(bytes, 0),
            Err(error) => panic!("expected zero byte size, got {error:?}"),
        }
    }

    #[test]
    fn aligned_size_overflow_is_allocation_failure() {
        match WgpuDevice::aligned_size(u64::MAX, wgpu::COPY_BUFFER_ALIGNMENT) {
            Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
                message,
                format!(
                    "buffer byte size {} cannot be aligned to {} bytes",
                    u64::MAX,
                    wgpu::COPY_BUFFER_ALIGNMENT
                )
            ),
            other => panic!("expected allocation failure, got {other:?}"),
        }
    }

    #[test]
    fn byte_size_overflow_is_allocation_failure() {
        let overflowing_len = usize::MAX / core::mem::size_of::<u64>() + 1;
        match WgpuDevice::byte_size::<u64>(overflowing_len) {
            Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
                message,
                format!("buffer length {overflowing_len} overflows byte size for 8-byte elements")
            ),
            other => panic!("expected allocation failure, got {other:?}"),
        }
    }
}
