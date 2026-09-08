# ADR 0059: Fallible WGPU storage allocation

- Status: Accepted
- Date: 2026-09-08
- Board item: [HEPH-WGPU-STORAGE-ALLOCATION](../../backlog.md#heph-wgpu-storage-allocation)

## Context

The storage allocation and upload implementations call WGPU buffer creation
without error scopes. WGPU reports validation, out-of-memory and internal
errors through scopes or the uncaptured-error handler. Returning a buffer from
the API therefore does not establish successful allocation. The upload
convenience API additionally expects mapped-range acquisition to succeed.
Odd byte lengths reserve host padding through an infallible vector allocation.

## Decision

Validate padded physical bytes against the acquired device's enabled
`max_buffer_size` before allocating a buffer or host padding. Preserve the
logical-length and zero-padding contract in [ADR 0008](0008-odd-length-wgpu-storage.md).
Reserve required host padding with `try_reserve_exact`.

Share storage allocation between empty/zeroed/uninitialized allocation and
upload. Capture all three WGPU error filters, pop in reverse nesting order,
and consume every scope before returning. Out-of-memory errors retain the
`AllocationFailed` category; validation and internal errors retain
`DispatchFailed` and the provider diagnostic. Enabled-size rejection reports
requested padded bytes and the actual enabled limit. Upload acquires its
mapped range through the fallible WGPU API, writes the complete payload, then
drops that view before unmapping. No partially initialized buffer is returned.

The locked WGPU 30.0.1 native implementation of `pop_error_scope` returns
`ready(scope.error)` (`src/backend/wgpu_core.rs`). This workspace disables
WGPU default features and enables only `std`, `wgsl`, `vulkan` and `metal`;
browser WebGPU is not enabled. Consume each future with one nonblocking poll.
An unexpectedly pending scope returns a typed dispatch failure rather than
introducing an unbounded wait. This synchronous native contract must be
revisited before enabling an asynchronous browser backend. Device readback
continues to use [ADR 0054](0054-bounded-default-device-waits.md) deadlines.

## Alternatives

- Wrapping `create_buffer_init` in scopes retains its internal mapped-range
  `expect`, so upload uses the underlying fallible mapped-range API.
- Adapter limits overstate what the acquired logical device enables.
- Unbounded `block_on` obscures the native synchronous completion contract.

## Verification and limits

Real-device tests request a 256-byte enabled limit, reject a 257-byte logical
payload padded to 260 bytes, and roundtrip a payload at the exact limit.
Generic exact-value tests cover empty, odd and aligned payloads across byte,
halfword, word, integer and shipped real scalar representations.
A real invalid buffer-usage descriptor exercises validation capture; a
subsequent successful upload checks that the scopes have been released.

These tests do not fabricate or prove physical device OOM recovery. Host
reservation retains the standard allocation diagnostic; the existing public
Hephaestus error contract stores provider diagnostics as text. This change
does not promise recovery from WGPU's internal host allocator aborts. Staging
pool allocation and transfer-write error scopes remain outside this item.

## Revision 2026-09-08: transfer preparation

[HEPH-WGPU-TRANSFER-ERRORS](../../backlog.md#heph-wgpu-transfer-errors)
extends the same buffer error boundary to staging/uniform allocation and
full/subrange queue writes. Pool misses validate the aligned physical size
against the enabled limit before allocation. A single descriptor allocator
serves storage, staging and uniform usage; every native creation and mapped
initialization operation uses the shared scope consumer.

Write destinations must belong to the calling device. Check the existing
buffer owner identity before calling WGPU: a foreign instance's resource ID
can panic during lookup before reaching scoped validation. Native queue
writes capture allocation, validation and internal errors. Length, offset
and interior-alignment rejection still occurs before a queue operation and
preserves destination values; successful tail writes may zero only physical
padding beyond the final logical value. This contract does not make a raw
WGPU queue escape hatch fallible or roll back earlier submitted work.

Locked WGPU 30.0.1 scopes are thread-specific, not one device-global stack:
`backend/wgpu_core.rs` stores scopes by `ThreadId` at line 634, routes errors
using the current thread at line 663, and pushes/pops that thread's stack at
lines 1847 and 1866. Native queue writes report errors synchronously through
that sink. The shared synchronous closure pushes, executes and consumes its
scopes on the same thread; no serialization lock is required. A barrier-based
real-descriptor test overlaps two complete scope stacks and asserts each
thread receives only its own diagnostic. Revisit this argument if WGPU changes
error delivery or this closure becomes asynchronous.

The public baseline run `c8ae1ee2` reproduces both the 264-byte staging request
against an enabled 256-byte limit and a cross-device write panic. Regressions
also exercise destroyed-buffer queue errors, valid transfers after rejection,
and exact retained bytes around accepted and rejected subranges. No physical
OOM, driver-fault injection, or performance guarantee is claimed.
