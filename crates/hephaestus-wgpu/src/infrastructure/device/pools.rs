//! Staging and uniform transient pools.

use super::WgpuDevice;
use crate::infrastructure::pool::{BufferOrigin, PooledBuffer};
use hephaestus_core::Result;
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl WgpuDevice {
    /// Retrieve a staging buffer of size >= size from the bounded pool, or
    /// create a new one. The size is automatically aligned to
    /// `wgpu::MAP_ALIGNMENT` (8 bytes).
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AllocationFailed`] when `size` cannot be aligned
    /// without overflowing `u64`, exceeds the enabled device buffer limit, or
    /// allocation fails. [`HephaestusError::DispatchFailed`] preserves scoped
    /// WGPU validation and internal errors.
    pub(crate) fn get_staging_buffer(&self, size: u64) -> Result<PooledBuffer> {
        let staging_size = Self::aligned_size(size, wgpu::MAP_ALIGNMENT)?;
        self.maybe_decay_staging();
        Ok(PooledBuffer::new(
            if let Some(buffer) = self.staging_pool.take_at_least(staging_size) {
                self.staging_accounting.record_hit(buffer.0.size());
                buffer.0
            } else {
                self.staging_accounting.record_miss();
                self.allocate_buffer(&wgpu::BufferDescriptor {
                    label: Some("hephaestus-recycled-staging"),
                    size: staging_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })?
            },
            BufferOrigin::Staging {
                pool: Arc::clone(&self.staging_pool),
                accounting: Arc::clone(&self.staging_accounting),
            },
        ))
    }

    /// Release parked staging retention when no staging transfer has happened
    /// for [`STAGING_POOL_IDLE_DECAY`]. Runs on the staging acquire path so
    /// decay needs no timer infrastructure; when warm it costs one clock read
    /// and two relaxed loads.
    fn maybe_decay_staging(&self) {
        let accounting = &self.staging_accounting;
        if accounting.idle_beyond(accounting.decay_deadline())
            && accounting.retained_bytes.load(Ordering::Relaxed) > 0
        {
            self.staging_pool.clear();
            accounting.reset_retained();
        }
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
    /// without overflowing `u64`, exceeds the enabled device buffer limit, or
    /// allocation fails. [`HephaestusError::DispatchFailed`] preserves scoped
    /// WGPU validation and internal errors.
    pub(crate) fn get_uniform_buffer(&self, size: u64) -> Result<PooledBuffer> {
        let uniform_size = Self::aligned_size(size, wgpu::COPY_BUFFER_ALIGNMENT)?;
        Ok(PooledBuffer::new(
            if let Some(buffer) = self.uniform_pool.take_at_least(uniform_size) {
                buffer.0
            } else {
                self.allocate_buffer(&wgpu::BufferDescriptor {
                    label: Some("hephaestus-recycled-uniform"),
                    size: uniform_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })?
            },
            BufferOrigin::Uniform {
                pool: Arc::clone(&self.uniform_pool),
            },
        ))
    }

    /// Drop transient staging and uniform buffers retained for reuse.
    ///
    /// The bounded pools avoid repeated staging and uniform allocations on hot
    /// paths. Bindings and short-lived host integrations can call this at an
    /// ownership boundary to release cached allocations before the host runtime
    /// tears down GPU state. The small fixed readback-completion slots remain
    /// device-owned because pending WGPU callbacks can still reference them.
    ///
    /// Transient allocations are provider-owned. Custom interop allocations
    /// created through [`Self::device`] remain caller-owned and cannot be
    /// inserted into these pools.
    ///
    /// ```compile_fail
    /// use hephaestus_wgpu::WgpuDevice;
    /// let owner = WgpuDevice::try_default("owner").unwrap();
    /// let donor = WgpuDevice::try_default("donor").unwrap();
    /// let foreign = donor.get_staging_buffer(8).unwrap();
    /// owner.recycle_staging_buffer(foreign);
    /// ```
    ///
    /// ```compile_fail
    /// use hephaestus_wgpu::WgpuDevice;
    /// let owner = WgpuDevice::try_default("owner").unwrap();
    /// let donor = WgpuDevice::try_default("donor").unwrap();
    /// let foreign = donor.get_uniform_buffer(4).unwrap();
    /// owner.recycle_uniform_buffer(foreign);
    /// ```
    #[inline]
    pub fn clear_transient_pools(&self) {
        self.staging_pool.clear();
        self.uniform_pool.clear();
        // The shadow retained bound must follow the release, or a later decay
        // check would see stale parking (harmless over-clear of an empty
        // pool, but avoidable).
        self.staging_accounting.reset_retained();
    }
}
