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

/// Generic RAII guard that recycles a wgpu buffer back to a pool on drop.
///
/// The recycle strategy `F` is a function that returns the buffer to its pool.
/// Callers use the type-aliased guards [`StagingBufferGuard`] and
/// [`UniformBufferGuard`] rather than instantiating this type directly.
///
/// This is the SSOT for all pooled-buffer RAII logic; both guard variants share
/// the identical fields, `Deref` impl, and `Drop` impl.
pub struct PoolBufferGuard<F>
where
    F: Fn(&WgpuDevice, wgpu::Buffer),
{
    device: WgpuDevice,
    buffer: Option<wgpu::Buffer>,
    recycle: F,
}

impl<F: Fn(&WgpuDevice, wgpu::Buffer)> PoolBufferGuard<F> {
    #[inline]
    #[must_use]
    pub(crate) fn new(device: WgpuDevice, buffer: wgpu::Buffer, recycle: F) -> Self {
        Self {
            device,
            buffer: Some(buffer),
            recycle,
        }
    }
}

impl<F: Fn(&WgpuDevice, wgpu::Buffer)> std::ops::Deref for PoolBufferGuard<F> {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("invariant: buffer is not dropped")
    }
}

impl<F: Fn(&WgpuDevice, wgpu::Buffer)> Drop for PoolBufferGuard<F> {
    #[inline]
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            (self.recycle)(&self.device, buffer);
        }
    }
}

/// RAII guard that automatically recycles a staging buffer back to the device's pool on drop.
pub type StagingBufferGuard = PoolBufferGuard<fn(&WgpuDevice, wgpu::Buffer)>;

/// RAII guard that automatically recycles a uniform buffer back to the device's pool on drop.
pub type UniformBufferGuard = PoolBufferGuard<fn(&WgpuDevice, wgpu::Buffer)>;

/// Construct a [`StagingBufferGuard`] — wraps a buffer that is returned to the
/// staging pool on drop.
#[inline]
#[must_use]
pub(crate) fn staging_guard(device: WgpuDevice, buffer: wgpu::Buffer) -> StagingBufferGuard {
    PoolBufferGuard::new(device, buffer, |d, b| d.recycle_staging_buffer(b))
}

/// Construct a [`UniformBufferGuard`] — wraps a buffer that is returned to the
/// uniform pool on drop.
#[inline]
#[must_use]
pub(crate) fn uniform_guard(device: WgpuDevice, buffer: wgpu::Buffer) -> UniformBufferGuard {
    PoolBufferGuard::new(device, buffer, |d, b| d.recycle_uniform_buffer(b))
}
