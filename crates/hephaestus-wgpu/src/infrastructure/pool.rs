//! Transient buffer pooling using Moirai's sharded resource pool.

use crate::infrastructure::device::WgpuDevice;

/// Zero-cost orphan-rule wrapper around wgpu::Buffer that implements SizeBounded.
#[repr(transparent)]
pub struct PoolBuffer(pub wgpu::Buffer);

impl moirai_sync::SizeBounded for PoolBuffer {
    #[inline]
    fn size(&self) -> u64 {
        self.0.size()
    }
}

impl std::ops::Deref for PoolBuffer {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// RAII guard that automatically recycles a staging buffer back to the device's pool on drop.
pub struct StagingBufferGuard {
    pub(crate) device: WgpuDevice,
    pub(crate) buffer: Option<wgpu::Buffer>,
}

impl StagingBufferGuard {
    #[inline]
    #[must_use]
    pub(crate) fn new(device: WgpuDevice, buffer: wgpu::Buffer) -> Self {
        Self {
            device,
            buffer: Some(buffer),
        }
    }
}

impl std::ops::Deref for StagingBufferGuard {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("invariant: buffer is not dropped")
    }
}

impl Drop for StagingBufferGuard {
    #[inline]
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.device.recycle_staging_buffer(buffer);
        }
    }
}

/// RAII guard that automatically recycles a uniform buffer back to the device's pool on drop.
pub struct UniformBufferGuard {
    pub(crate) device: WgpuDevice,
    pub(crate) buffer: Option<wgpu::Buffer>,
}

impl UniformBufferGuard {
    #[inline]
    #[must_use]
    pub(crate) fn new(device: WgpuDevice, buffer: wgpu::Buffer) -> Self {
        Self {
            device,
            buffer: Some(buffer),
        }
    }
}

impl std::ops::Deref for UniformBufferGuard {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("invariant: buffer is not dropped")
    }
}

impl Drop for UniformBufferGuard {
    #[inline]
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.device.recycle_uniform_buffer(buffer);
        }
    }
}
