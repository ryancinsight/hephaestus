//! Transient buffer and readback-completion pooling.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use hephaestus_core::{HephaestusError, Result};

use crate::infrastructure::device::WgpuDevice;

const MAP_PENDING: u8 = 0;
const MAP_SUCCEEDED: u8 = 1;
const MAP_FAILED: u8 = 2;

/// Reusable completion state for one asynchronous WGPU buffer mapping.
#[derive(Clone, Debug)]
pub(crate) struct MapCompletion {
    state: Arc<AtomicU8>,
}

impl MapCompletion {
    fn new() -> Self {
        #[cfg(test)]
        MAP_COMPLETION_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        Self {
            state: Arc::new(AtomicU8::new(MAP_PENDING)),
        }
    }

    pub(crate) fn reset(&self) {
        self.state.store(MAP_PENDING, Ordering::Relaxed);
    }

    pub(crate) fn complete(&self, result: core::result::Result<(), wgpu::BufferAsyncError>) {
        self.state.store(
            if result.is_ok() {
                MAP_SUCCEEDED
            } else {
                MAP_FAILED
            },
            Ordering::Release,
        );
    }

    pub(crate) fn result(&self) -> Result<()> {
        match self.state.load(Ordering::Acquire) {
            MAP_SUCCEEDED => Ok(()),
            MAP_FAILED => Err(HephaestusError::TransferFailed {
                message: "buffer mapping failed".to_owned(),
            }),
            _ => Err(HephaestusError::TransferFailed {
                message: "map_async callback was not delivered after device completion".to_owned(),
            }),
        }
    }
}

/// Bounded retained completion slots for synchronous readback callbacks.
///
/// Acquiring and returning a slot holds the mutex only for a vector pop or
/// push. Each in-flight mapping owns an independent atomic state, so readbacks
/// remain concurrent. Concurrency above the retained capacity allocates an
/// unpooled overflow slot rather than serializing device work.
#[derive(Debug)]
pub(crate) struct MapCompletionPool {
    available: Mutex<Vec<MapCompletion>>,
    retained_capacity: usize,
}

impl MapCompletionPool {
    pub(crate) fn new(retained_capacity: usize) -> Self {
        let mut available = Vec::with_capacity(retained_capacity);
        available.resize_with(retained_capacity, MapCompletion::new);
        Self {
            available: Mutex::new(available),
            retained_capacity,
        }
    }

    pub(crate) fn take(&self) -> MapCompletion {
        self.available
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_else(MapCompletion::new)
    }

    pub(crate) fn recycle(&self, completion: MapCompletion) {
        let mut available = self
            .available
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if available.len() < self.retained_capacity {
            available.push(completion);
        }
    }

    pub(crate) fn clear(&self) {
        self.available
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}

#[cfg(test)]
static MAP_COMPLETION_ALLOCATIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
pub(crate) fn map_completion_allocation_count() -> u64 {
    MAP_COMPLETION_ALLOCATIONS.load(Ordering::Relaxed)
}

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
