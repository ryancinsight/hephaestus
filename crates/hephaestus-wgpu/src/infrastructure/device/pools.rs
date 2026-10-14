use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use hephaestus_core::Result;

use crate::infrastructure::pool::{BufferOrigin, PooledBuffer};

use super::WgpuDevice;

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
pub(super) const STAGING_POOL_MAX_BUFFERS: usize = 8;
pub(super) const STAGING_POOL_MAX_BYTES: u64 = 512 * 1024 * 1024;
// Idle decay for the retention ceiling above: the pool parks burst
// allocations until `clear_transient_pools` is called, which in a long-lived
// session would keep hundreds of MiB of MAP_READ memory for the device's
// lifetime after a single readback burst. Staging retention is instead
// released once no staging traffic (acquire or recycle) has happened for
// `STAGING_POOL_IDLE_DECAY`; the check rides the next acquire (one monotonic
// clock read, two relaxed loads), so there is no timer thread and no added
// cost on a busy path.
//
// The 10 s deadline separates two boundaries. Within an active run, staging
// traffic recurs every solver step and routine inter-readback gaps are
// milliseconds — three-plus orders of magnitude below the deadline — so a
// warm pool is never decayed mid-run (the sustained-traffic test pins the
// hit-rate contract). Across a session boundary (paused, finished, one-shot
// validation), the deadline bounds parked MAP_READ memory to ~10 s past the
// last transfer, reclaiming the burst within a human-noticeable idle window.
// Uniforms are exempt: their whole retention is ~8 KiB and every dispatch
// reuses it, so decaying them would spend a fresh allocation on a
// non-problem.
const STAGING_POOL_IDLE_DECAY: Duration = Duration::from_secs(10);

/// Idle-decay accounting for the staging pool.
///
/// `ShardedResourcePool`'s retention counters are private, so this shadow
/// tracks the bytes routed through this device's staging acquire/recycle
/// paths. The retained total is an UPPER BOUND on true pool residency: the
/// pool's internal FIFO cap eviction and over-cap recycles drop buffers this
/// shadow still counts. That is the safe direction for a decay decision — it
/// can only trigger a `clear` of space that is (partly) already gone, never
/// retain beyond the policy.
#[derive(Debug)]
pub(crate) struct StagingPoolAccounting {
    /// Monotonic clock base; `last_use_ms` is an offset from this.
    base: Instant,
    last_use_ms: AtomicU64,
    /// Upper bound on bytes parked in the staging pool (see type doc).
    retained_bytes: AtomicU64,
    /// Acquires served from the pool versus fresh buffer creations.
    pub(super) hits: AtomicU64,
    pub(super) misses: AtomicU64,
    #[cfg(test)]
    /// Test-only decay deadline override in ms; 0 selects
    /// [`STAGING_POOL_IDLE_DECAY`]. Lets unit tests exercise decay without
    /// waiting out the production deadline.
    pub(super) test_decay_ms: AtomicU64,
    /// Test-only monotonic clock value in ms; 0 reads the real clock.
    #[cfg(test)]
    pub(super) test_now_ms: AtomicU64,
}

impl StagingPoolAccounting {
    pub(super) fn new() -> Self {
        Self {
            base: Instant::now(),
            last_use_ms: AtomicU64::new(0),
            retained_bytes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            #[cfg(test)]
            test_decay_ms: AtomicU64::new(0),
            #[cfg(test)]
            test_now_ms: AtomicU64::new(0),
        }
    }

    #[inline]
    fn now_ms(&self) -> u64 {
        #[cfg(test)]
        {
            let test_now_ms = self.test_now_ms.load(Ordering::Relaxed);
            if test_now_ms != 0 {
                return test_now_ms;
            }
        }
        self.base.elapsed().as_millis() as u64
    }

    /// Record staging traffic, keeping pool retention warm.
    #[inline]
    fn touch(&self) {
        self.last_use_ms.store(self.now_ms(), Ordering::Relaxed);
    }

    /// True when no staging traffic happened within `deadline`.
    #[inline]
    fn idle_beyond(&self, deadline: Duration) -> bool {
        self.now_ms()
            .saturating_sub(self.last_use_ms.load(Ordering::Relaxed))
            > deadline.as_millis() as u64
    }

    /// Decay deadline: the test-only override when one is set, otherwise the
    /// production constant.
    fn decay_deadline(&self) -> Duration {
        #[cfg(test)]
        {
            let override_ms = self.test_decay_ms.load(Ordering::Relaxed);
            if override_ms != 0 {
                return Duration::from_millis(override_ms);
            }
        }
        STAGING_POOL_IDLE_DECAY
    }

    pub(crate) fn record_recycle(&self, bytes: u64) {
        self.retained_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.touch();
    }

    fn record_hit(&self, bytes: u64) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.retained_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.touch();
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    fn reset_retained(&self) {
        self.retained_bytes.store(0, Ordering::Relaxed);
    }
}
// Uniforms: metadata blocks of ≲256 B; ops acquire up to three per call
// (`matmul_into`, `kron_into`), so 2/shard (the old 8/4) forced an
// allocate-evict cycle on every 3-uniform call from one thread. 32 (8/shard)
// retains at most ~8 KiB of uniforms per shard against the 1 MiB/4 budget.
pub(super) const UNIFORM_POOL_MAX_BUFFERS: usize = 32;
pub(super) const UNIFORM_POOL_MAX_BYTES: u64 = 1024 * 1024;

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
