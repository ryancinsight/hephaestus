use core::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use std::any::TypeId;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::pool::BoundedBufferPool;

/// Pipeline-cache key: kernel-family discriminator, scalar type, block width.
pub(crate) type PipelineKey = (TypeId, TypeId, u32);
pub(crate) type PipelineCache =
    Arc<moirai_sync::sync::ConcurrentHashMap<PipelineKey, wgpu::ComputePipeline>>;

const TRANSIENT_POOL_MAX_BUFFERS: usize = 8;
const STAGING_POOL_MAX_BYTES: u64 = 64 * 1024 * 1024;
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
    topology: Option<Arc<themis::GpuTopology>>,
    pub(crate) pipeline_cache: PipelineCache,
    pub(crate) staging_pool: Arc<Mutex<BoundedBufferPool<wgpu::Buffer>>>,
    pub(crate) uniform_pool: Arc<Mutex<BoundedBufferPool<wgpu::Buffer>>>,
}

impl WgpuDevice {
    /// Wrap an existing device and queue.
    ///
    /// No adapter is available on this path, so no topology snapshot is
    /// reported ([`topology`](Self::topology) returns `None`); the
    /// `try_default*` acquisition paths capture one from the adapter.
    #[must_use]
    #[inline]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            topology: None,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            staging_pool: Arc::new(Mutex::new(BoundedBufferPool::new(
                TRANSIENT_POOL_MAX_BUFFERS,
                STAGING_POOL_MAX_BYTES,
            ))),
            uniform_pool: Arc::new(Mutex::new(BoundedBufferPool::new(
                TRANSIENT_POOL_MAX_BUFFERS,
                UNIFORM_POOL_MAX_BYTES,
            ))),
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
        let try_acquire = |instance: &wgpu::Instance| -> Option<Self> {
            let try_device = |adapter: &wgpu::Adapter| -> std::result::Result<
                (wgpu::Device, wgpu::Queue),
                wgpu::RequestDeviceError,
            > {
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some(label),
                    required_features,
                    required_limits: required_limits.clone(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                }))
            };

            // Try High Performance hardware adapter first
            if let Ok(adapter) =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
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

            // Fallback to software/fallback adapter
            if let Ok(fallback_adapter) =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                }))
            {
                let topology = Self::topology_from_adapter(&fallback_adapter);
                if let Ok((device, queue)) = try_device(&fallback_adapter) {
                    let mut acquired = Self::new(Arc::new(device), Arc::new(queue));
                    acquired.topology = Some(Arc::new(topology));
                    return Some(acquired);
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
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some(label),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                }))
            };

            if let Ok(adapter) =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
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

        let mut pool = self
            .staging_pool
            .lock()
            .expect("invariant: staging pool mutex is not poisoned");
        Ok(if let Some(buffer) = pool.take_at_least(staging_size) {
            buffer
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hephaestus-recycled-staging"),
                size: staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Return a staging buffer back to the bounded pool for reuse.
    pub fn recycle_staging_buffer(&self, buffer: wgpu::Buffer) {
        let mut pool = self
            .staging_pool
            .lock()
            .expect("invariant: staging pool mutex is not poisoned");
        pool.recycle(buffer);
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

        let mut pool = self
            .uniform_pool
            .lock()
            .expect("invariant: uniform pool mutex is not poisoned");
        Ok(if let Some(buffer) = pool.take_at_least(uniform_size) {
            buffer
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hephaestus-recycled-uniform"),
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Return a uniform buffer back to the bounded pool for reuse.
    pub fn recycle_uniform_buffer(&self, buffer: wgpu::Buffer) {
        let mut pool = self
            .uniform_pool
            .lock()
            .expect("invariant: uniform pool mutex is not poisoned");
        pool.recycle(buffer);
    }

    /// Drop transient buffers retained for reuse.
    ///
    /// The bounded pools avoid repeated staging and uniform allocations on hot
    /// paths. Bindings and short-lived host integrations can call this at an
    /// ownership boundary to release cached allocations before the host runtime
    /// tears down GPU state.
    #[inline]
    pub fn clear_transient_pools(&self) {
        self.staging_pool
            .lock()
            .expect("invariant: staging pool mutex is not poisoned")
            .clear();
        self.uniform_pool
            .lock()
            .expect("invariant: uniform pool mutex is not poisoned")
            .clear();
    }
}

impl ComputeDevice for WgpuDevice {
    type Buffer<T: Pod> = WgpuBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "wgpu"
    }

    fn alloc_zeroed<T: Pod>(&self, len: usize) -> Result<WgpuBuffer<T>> {
        // WebGPU guarantees newly created buffers are zero-initialized.
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hephaestus-storage"),
            size: Self::padded_size::<T>(len)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(WgpuBuffer {
            buffer,
            len,
            marker: PhantomData,
        })
    }

    fn upload<T: Pod>(&self, host: &[T]) -> Result<WgpuBuffer<T>> {
        let byte_len = Self::byte_size::<T>(host.len())?;
        let padded_len = Self::padded_size::<T>(host.len())?;
        let buffer = if padded_len == 0 {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hephaestus-upload"),
                size: 0,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            let mut padded = vec![0u8; padded_len as usize];
            padded[..byte_len as usize].copy_from_slice(bytemuck::cast_slice(host));
            (*self.device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hephaestus-upload"),
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(WgpuBuffer {
            buffer,
            len: host.len(),
            marker: PhantomData,
        })
    }

    fn download<T: Pod>(&self, buffer: &WgpuBuffer<T>, out: &mut [T]) -> Result<()> {
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
        let staging = self.get_staging_buffer(padded)?;
        let staging_size = staging.size();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-download"),
            });
        encoder.copy_buffer_to_buffer(&buffer.buffer, 0, &staging, 0, padded);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..staging_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
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

        let mapped = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&mapped[..byte_len as usize]));
        drop(mapped);
        staging.unmap();

        self.recycle_staging_buffer(staging);

        Ok(())
    }

    fn write_buffer<T: Pod>(&self, buffer: &WgpuBuffer<T>, host: &[T]) -> Result<()> {
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
