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
    pub(crate) pipeline_cache: Arc<Mutex<HashMap<(TypeId, TypeId), wgpu::ComputePipeline>>>,
    pub(crate) staging_pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
}

impl WgpuDevice {
    /// Wrap an existing device and queue.
    #[must_use]
    #[inline]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            staging_pool: Arc::new(Mutex::new(Vec::new())),
        }
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
        Ok(Self::new(Arc::new(device), Arc::new(queue)))
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
