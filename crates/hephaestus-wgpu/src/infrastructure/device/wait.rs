//! Bounded device waits and typed poll-failure classification.

use hephaestus_core::HephaestusError;
use std::time::Duration;

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
pub(super) const DEFAULT_DEVICE_WAIT: Duration = Duration::from_secs(30);

#[cfg(test)]
thread_local! {
    /// Test-only deadline override in nanoseconds; 0 selects
    /// [`DEFAULT_DEVICE_WAIT`]. The production bound cannot be reached inside
    /// the configured test budget — doing so needs 30 s of stalled device — so
    /// the override lets a test drive a real submission that is genuinely too
    /// slow for its deadline into the timeout path, rather than asserting the
    /// plumbing against itself. Thread-local, so a case running concurrently
    /// on another thread keeps the production bound.
    pub(super) static TEST_WAIT_DEADLINE_NS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
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
