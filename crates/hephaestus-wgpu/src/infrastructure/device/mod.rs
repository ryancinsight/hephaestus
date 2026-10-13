mod acquisition;
mod allocation;
#[cfg(test)]
mod pool_tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use hephaestus_core::{HephaestusError, Result};
use std::any::TypeId;

use crate::infrastructure::pool::{MapCompletionPool, PoolBuffer};
use moirai_sync::ShardedResourcePool;

#[cfg(test)]
use hephaestus_core::{ComputeDevice, DeviceFeature, DevicePreference};

/// Pipeline-cache key: kernel-family discriminator, scalar type, block width.
pub(crate) type PipelineKey = (TypeId, TypeId, u32);
/// Collision-safe key for runtime-generated fusion pipelines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FusionPipelineKey {
    pub(crate) family: TypeId,
    pub(crate) scalar: TypeId,
    pub(crate) rank: u32,
    pub(crate) input_count: u32,
    pub(crate) expression: Arc<str>,
}
pub(crate) type PipelineCache = Arc<
    moirai_sync::sync::ConcurrentHashMap<
        PipelineKey,
        Arc<std::sync::OnceLock<wgpu::ComputePipeline>>,
    >,
>;
pub(crate) type FusionPipelineCache = Arc<
    moirai_sync::sync::ConcurrentHashMap<
        FusionPipelineKey,
        Arc<std::sync::OnceLock<std::result::Result<wgpu::ComputePipeline, Arc<str>>>>,
    >,
>;

/// Cancels a pending map or releases an active mapping before pooled staging
/// storage can be recycled, including error and unwind exits.
struct MappingLifecycle<'a>(&'a wgpu::Buffer);

#[derive(Clone, Copy)]
struct ReadbackRegion {
    byte_offset: u64,
    padded: u64,
    byte_len: u64,
}

impl Drop for MappingLifecycle<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.unmap();
    }
}

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
pub(super) struct StagingPoolAccounting {
    /// Monotonic clock base; `last_use_ms` is an offset from this.
    base: Instant,
    last_use_ms: AtomicU64,
    /// Upper bound on bytes parked in the staging pool (see type doc).
    retained_bytes: AtomicU64,
    /// Acquires served from the pool versus fresh buffer creations.
    hits: AtomicU64,
    misses: AtomicU64,
    #[cfg(test)]
    /// Test-only decay deadline override in ms; 0 selects
    /// [`STAGING_POOL_IDLE_DECAY`]. Lets unit tests exercise decay without
    /// waiting out the production deadline.
    test_decay_ms: AtomicU64,
    /// Test-only monotonic clock value in ms; 0 reads the real clock.
    #[cfg(test)]
    test_now_ms: AtomicU64,
}

impl StagingPoolAccounting {
    fn new() -> Self {
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

    pub(super) fn record_recycle(&self, bytes: u64) {
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
const UNIFORM_POOL_MAX_BUFFERS: usize = 32;
const UNIFORM_POOL_MAX_BYTES: u64 = 1024 * 1024;

// Deadline carried by every device wait this backend performs on a default
// path: readback polls (`download`, `download_owned`, `download_sub_buffer`,
// the decomposition region readbacks), `copy_buffer`, and `synchronize`. An
// unbounded wait is the temporal form of an unbounded queue — a wedged driver
// or a lost device parks the calling thread, which here is a Python binding or
// a solver step, with nothing to act on.
//
// DERIVATION. The bound sits ABOVE the platform's own hang detection, so this
// deadline is the backstop rather than the first reporter:
//   * Windows resets a GPU that misses `TdrDelay` (default 2 s) and allows the
//     driver `TdrDdiDelay` (default 5 s) to unwind before bug-checking, so a
//     genuine hang surfaces through wgpu as a lost device within ~7 s.
//     Source: Microsoft, "Testing and Debugging TDR During Driver Development"
//     (TDR registry keys) — TdrDelay 2 s, TdrDdiDelay 5 s, TdrLimitTime 60 s.
//   * Linux amdgpu's GPU-scheduler `lockup_timeout` defaults to 2000 ms across
//     all queues, inside the same envelope.
// A shorter deadline would fire first and replace an accurate device-lost
// diagnosis with an ambiguous host-side timeout. 30 s is ~4x the 7 s envelope,
// so it fires only where the platform watchdog does not: a compute stack with
// no watchdog, a wedged driver, or work legitimately queued ahead of the
// waited submission for longer than the bound.
//
// UPPER BOUND. 30 s is half of Windows' `TdrLimitTime` (60 s) bug-check window
// and remains a host stall a human notices. Headroom against real work: this
// backend's whole 172-case integration contract suite, every case
// device-resident, completes in ~1.9 s of wall clock, so no wait this backend
// produces is within an order of magnitude of the bound. Callers whose work
// genuinely exceeds it wait explicitly through `download_with_timeout` or
// `WgpuCommandStream::submit_with_timeout` instead of being silently unbounded.
const DEFAULT_DEVICE_WAIT: Duration = Duration::from_secs(30);

#[cfg(test)]
thread_local! {
    /// Test-only deadline override in nanoseconds; 0 selects
    /// [`DEFAULT_DEVICE_WAIT`]. The production bound cannot be reached inside
    /// the configured test budget — doing so needs 30 s of stalled device — so
    /// the override lets a test drive a real submission that is genuinely too
    /// slow for its deadline into the timeout path, rather than asserting the
    /// plumbing against itself. Thread-local, so a case running concurrently
    /// on another thread keeps the production bound.
    static TEST_WAIT_DEADLINE_NS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// The deadline every default device wait in this backend carries.
pub(crate) fn device_wait_deadline() -> Duration {
    #[cfg(test)]
    {
        let override_ns = TEST_WAIT_DEADLINE_NS.with(core::cell::Cell::get);
        if override_ns != 0 {
            return Duration::from_nanos(override_ns);
        }
    }
    DEFAULT_DEVICE_WAIT
}

/// Map a wgpu poll failure onto the typed error a caller can act on.
///
/// An elapsed deadline is not a transfer fault, so it carries its own variant
/// and the bound that expired; every other poll failure stays a transfer
/// failure. Nothing here retries, degrades, or falls back — the caller decides.
pub(crate) fn poll_failure(
    context: &str,
    deadline: Duration,
    error: &wgpu::PollError,
) -> HephaestusError {
    match error {
        wgpu::PollError::Timeout => HephaestusError::DeviceWaitTimeout {
            deadline,
            message: context.to_string(),
        },
        other @ wgpu::PollError::WrongSubmissionIndex(..) => HephaestusError::TransferFailed {
            message: format!("{context}: device poll failed: {other:?}"),
        },
    }
}

/// An acquired wgpu device + queue pair.
///
/// Clones share the device, caches and transient pools. This is the authoritative
/// adapter/device acquisition for Atlas wgpu consumers; apollo's
/// `apollo-wgpu-helpers` delegates here instead of carrying its own copy.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: Option<wgpu::AdapterInfo>,
    adapter_features: Option<wgpu::Features>,
    adapter_limits: Option<wgpu::Limits>,
    topology: Option<Arc<themis::GpuTopology>>,
    pub(crate) pipeline_cache: PipelineCache,
    pub(crate) fusion_pipeline_cache: FusionPipelineCache,
    staging_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    uniform_pool: Arc<ShardedResourcePool<PoolBuffer>>,
    map_completion_pool: Arc<MapCompletionPool>,
    staging_accounting: Arc<StagingPoolAccounting>,
}

mod accessors;
mod construction;
mod constructors;
mod features;
mod pools;
mod provider;
#[cfg(test)]
mod readback_tests;
#[cfg(test)]
mod tests;
mod transfer;
