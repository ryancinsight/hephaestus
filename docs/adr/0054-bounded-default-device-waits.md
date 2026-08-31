# ADR 0054 — Bounded default device waits

- Status: Accepted
- Date: 2026-08-31
- Refs: `backlog.md#heph-wgpu-default-deadlines`; the WGPU default wait paths
  `download`, `download_owned`, `download_sub_buffer`, `copy_buffer`,
  `synchronize`, and the decomposition region readback.

## Context

The WGPU backend's default wait paths polled the device with `timeout: None`
(`wgpu::PollType::wait_indefinitely()`). Every one of them is an external
interaction — the host blocks until a foreign scheduler answers — and none
carried a deadline, so a lost device, a wedged driver, or a compute stack
without a watchdog parks the calling thread permanently. That thread is a
Python binding or a solver step; there is nothing for the caller to act on and
no way back.

Bounded forms already existed (`download_with_timeout`,
`WgpuCommandStream::submit_with_timeout`), but nothing routed a default through
one, so the bounded path was opt-in and the unbounded path was what shipped.

## Decision

**One backend-wide deadline constant, applied to every default wait, with an
elapsed deadline surfacing as its own error variant.**

1. `DEFAULT_DEVICE_WAIT = 30 s`, derived below, in
   `infrastructure/device.rs`. `device_wait_deadline()` is the single read
   point; `stage_and_read` and `download_into` take `Duration`, not
   `Option<Duration>`, so an unbounded wait is no longer expressible on these
   paths.
2. `HephaestusError::DeviceWaitTimeout { deadline, message }` in
   `hephaestus-core`. An elapsed deadline is not a transfer fault: nothing
   reported an error, the device did not answer. A caller acts on the two
   differently — a transfer failure is terminal for that operation, an elapsed
   deadline names an unresponsive or over-subscribed device that can be
   reacquired or re-submitted against. Collapsing it into `TransferFailed`
   would leave that decision to string matching, which the failure-mode
   preservation rule forbids. Every other `wgpu::PollError` stays
   `TransferFailed`.
3. **No new opt-out surface.** `download_with_timeout` and
   `submit_with_timeout` already serve callers whose bound differs; no caller
   in this workspace or the stack needs a different bound on `synchronize`,
   `download_owned`, or `copy_buffer`, so no parameter is added for one.

Nothing retries, degrades, or falls back. The timeout propagates.

### Value derivation

The bound sits **above** the platform's own hang detection, so it is the
backstop rather than the first reporter.

- Windows resets a GPU that misses `TdrDelay` (default **2 s**) and allows the
  driver `TdrDdiDelay` (default **5 s**) to unwind before bug-checking, so a
  genuine hang surfaces through wgpu as a lost device within **~7 s**. Source:
  Microsoft, *Testing and Debugging TDR During Driver Development* (TDR
  registry keys) — `TdrDelay` 2 s, `TdrDdiDelay` 5 s, `TdrLimitTime` 60 s.
- Linux amdgpu's GPU-scheduler `lockup_timeout` defaults to **2000 ms** across
  all queues, inside the same envelope. Source: kernel.org, amdgpu module
  parameters.

A deadline below ~7 s would fire first and replace an accurate device-lost
diagnosis with an ambiguous host-side timeout. 30 s is ~4× that envelope, so it
fires only where the platform watchdog does not: a compute-only stack with no
watchdog, a wedged driver, or work legitimately queued ahead of the waited
submission for longer than the bound.

Upper bound: 30 s is half of Windows' `TdrLimitTime` (60 s) bug-check window
and remains a host stall a human notices. Headroom against real work: this
backend's whole 172-case integration contract suite, every case
device-resident, completes in ~1.9 s of wall clock, so no wait this backend
produces is within an order of magnitude of the bound.

## Alternatives rejected

- **Keep `TransferFailed` with a timeout message.** Stringly-typed
  discrimination of a distinct failure mode; the caller cannot branch on it.
- **A configurable per-device deadline.** A configuration surface with no
  caller. The two existing per-call bounded forms already cover a differing
  bound.
- **A bound below the TDR envelope (e.g. 5 s).** Would pre-empt the platform's
  own recovery and report the less accurate cause.
- **Bounding `checked_submit`'s error-scope waits too.** Out of scope: that
  path does not poll, it awaits wgpu error-scope futures through
  `moirai::block_on`. Whether those can stall unboundedly is a separate
  question, recorded on the board rather than guessed at here.

## Consequences

- Adding a variant to the public `HephaestusError` is a breaking change under
  `cargo-semver-checks` (`enum_variant_added`), so the item reclassifies
  **[minor] → [major]** for the 0.x line. Every match on `HephaestusError` in
  this workspace and in the stack's consumers (apollo) names specific variants
  under a catch-all, so the addition is source-compatible with all of them; no
  consumer migration is required beyond the version bump at release.
- `#[non_exhaustive]` on `HephaestusError` is **not** taken here. It would make
  future variants non-breaking, but it is an API-governance decision over every
  backend's consumers rather than part of this bounded change, and it does not
  affect the correctness of this one.
- A timeout on a readback unmaps its staging allocation on the way out
  (`MappingLifecycle` is constructed before the poll), aborting the pending
  `map_async` rather than recycling a buffer with a live mapping request. The
  device stays usable, which the contract case asserts by downloading
  successfully after a deliberately overrun wait.

## Verification

- `infrastructure::device::tests::an_overrun_default_wait_reports_a_typed_timeout_and_recovers`
  drives the deadline to 1 ns behind 512 MiB of queued device-to-device copy
  traffic and asserts `download_owned` — the *default* path, no timeout
  argument — returns `DeviceWaitTimeout` carrying the elapsed deadline, then
  that the same path succeeds afterwards. The deadline is driven down rather
  than the device made to hang: a hang reachable from a test is either a 30 s
  stall (outside the configured nextest budget) or an infinite kernel (a TDR
  the host must recover from). The submission is real and genuinely slower than
  its deadline by five orders of magnitude, so the outcome does not depend on
  scheduling.
- Liveness proved twice. With the default path ignoring the deadline the case
  fails with `got Ok([7, 8, 9, 10])`; with `PollError::Timeout` mapped back to
  `TransferFailed` it fails with
  `got Err(TransferFailed { .. })`. It is the only case that fails in either
  experiment.
- `contract::bounded_default_waits_leave_the_success_path_unchanged` holds the
  other half over the public surface: with the production bound in force, all
  five newly-bounded entry points return exactly what they returned unbounded.
