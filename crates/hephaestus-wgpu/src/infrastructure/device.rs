use core::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use wgpu::util::DeviceExt;

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::infrastructure::buffer::WgpuBuffer;

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
    pub(crate) pipeline_cache: Arc<Mutex<HashMap<(TypeId, TypeId), wgpu::ComputePipeline>>>,
    pub(crate) staging_pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
    pub(crate) uniform_pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
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
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            staging_pool: Arc::new(Mutex::new(Vec::new())),
            uniform_pool: Arc::new(Mutex::new(Vec::new())),
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
    pub fn try_default_with_limits(label: &str, required_limits: wgpu::Limits) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| HephaestusError::AdapterUnavailable {
            message: e.to_string(),
        })?;
        let topology = Self::topology_from_adapter(&adapter);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| HephaestusError::DeviceUnavailable {
            message: e.to_string(),
        })?;
        let mut acquired = Self::new(Arc::new(device), Arc::new(queue));
        acquired.topology = Some(Arc::new(topology));
        Ok(acquired)
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

    /// Size in bytes of `len` elements of `T`, padded to wgpu copy alignment.
    fn padded_size<T>(len: usize) -> u64 {
        let raw = (len * core::mem::size_of::<T>()) as u64;
        raw.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT
    }

    /// Retrieve a staging buffer of size >= size from the pool, or create a new one.
    /// The size is automatically aligned to `wgpu::MAP_ALIGNMENT` (8 bytes).
    #[must_use]
    pub fn get_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        let staging_size = size.div_ceil(8) * 8;
        let mut pool = self.staging_pool.lock().unwrap();
        if let Some(pos) = pool.iter().position(|b| b.size() >= staging_size) {
            pool.swap_remove(pos)
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hephaestus-recycled-staging"),
                size: staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }
    }

    /// Return a staging buffer back to the pool for reuse.
    pub fn recycle_staging_buffer(&self, buffer: wgpu::Buffer) {
        let mut pool = self.staging_pool.lock().unwrap();
        pool.push(buffer);
    }

    /// Retrieve a uniform buffer of size ≥ `size` from the pool, or create
    /// one. Contents are written with `queue.write_buffer`, which is ordered
    /// on the queue timeline relative to submissions, so a recycled uniform
    /// can be rewritten for the next dispatch without racing in-flight work
    /// on the same queue.
    #[must_use]
    pub fn get_uniform_buffer(&self, size: u64) -> wgpu::Buffer {
        let uniform_size = size.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT;
        let mut pool = self.uniform_pool.lock().unwrap();
        if let Some(pos) = pool.iter().position(|b| b.size() >= uniform_size) {
            pool.swap_remove(pos)
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hephaestus-recycled-uniform"),
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }
    }

    /// Return a uniform buffer back to the pool for reuse.
    pub fn recycle_uniform_buffer(&self, buffer: wgpu::Buffer) {
        let mut pool = self.uniform_pool.lock().unwrap();
        pool.push(buffer);
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
            size: Self::padded_size::<T>(len),
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
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hephaestus-upload"),
                contents: bytemuck::cast_slice(host),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
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

        let byte_len = (buffer.len * core::mem::size_of::<T>()) as u64;
        let padded = Self::padded_size::<T>(buffer.len);
        let staging = self.get_staging_buffer(padded);
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
}
