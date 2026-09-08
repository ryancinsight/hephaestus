# ADR 0036: Keep accelerator COW copies on-device

- Status: Accepted
- Date: 2026-07-28
- Scope: `ComputeDevice` whole-buffer copies and Coeus Hephaestus storage
  uniqueness
- Change class: `[arch]`/`[patch]`

## Context

`StorageMut::make_unique` must detach a shared accelerator buffer before a
mutation. The existing Hephaestus implementation downloaded the source into a
temporary host `Vec` and uploaded it into a new device buffer. That path
performed two full transfers, allocated host memory proportional to tensor
size, and discarded the source buffer's memory-tier hint.

Hephaestus command streams already provide native whole-buffer copies for WGPU,
CUDA, and ROCm; Metal uses the WGPU stream selected with the native Metal
adapter. The device contract lacked a synchronous operation for consumers that
need to detach storage without exposing a command-stream implementation.

## Decision

Add `ComputeDevice::copy_buffer`, with equal-length typed buffers and a
completion-before-return contract. Each provider delegates to its existing
device-local command-stream copy and synchronizes before returning. Metal
delegates to its wrapped WGPU device, preserving one copy implementation for
the native Metal path. The unavailable CUDA and ROCm configurations return
their existing typed adapter error.

Coeus COW allocates the replacement with the source buffer's `MemoryTier` and
uses `copy_buffer`. No host allocation, host transfer, CPU fallback, or
provider-specific duplicate kernel is introduced.

## Alternatives rejected

- Retain the host round-trip: rejected because it adds O(n) host allocation
  and two full transfers to every shared-storage mutation.
- Expose provider-specific copy methods: rejected because it forks the
  consumer contract and forces COW to know each vendor API.
- Make the copy asynchronous: rejected because `make_unique` returns a
  synchronous storage mutation boundary and callers may immediately read or
  overwrite the detached buffer.

## Verification

The core trait and all WGPU, CUDA, ROCm, Metal, and unavailable-provider
implementations compile under their existing feature matrix. Coeus storage
uniqueness tests preserve value semantics while the source remains shared;
provider command-stream tests remain the native copy oracle. Runtime allocation
and transfer deltas require a controlled backend benchmark and are not claimed
by this structural change alone.

## Revisit trigger

Revisit if a provider can prove an equivalent completion contract without a
device-wide synchronization, or if a measured workload shows synchronization
dominates COW mutation after host staging is removed.

## Revision 2026-09-08: physical padding and prefix boundaries

[HEPH-WGPU-BUFFER-EXTENTS](../../backlog.md#heph-wgpu-buffer-extents)
applies [ADR 0008](0008-odd-length-wgpu-storage.md) to WGPU copies and clears
for scalar buffers whose byte lengths are not multiples of four. Whole-buffer
operations include physical allocation padding, which contains no other logical
values. Prefix copies transfer aligned words
and merge the final one to three bytes on-device, preserving destination bytes
outside the requested prefix. Two four-byte scratch buffers keep this tail
operation independent of full-buffer storage-binding limits. Tail allocation and
pipeline/binding preparation precede all prefix encoding so preparation errors
leave no partial copy in the command stream; no payload reaches
the host. Cached tail pipelines specialize only the three possible byte masks.

Rounding a prefix transfer upward is rejected because it overwrites logical
suffix values. The regression covers exact values over empty, odd, and aligned
lengths and every prefix boundary, including length rejection with unchanged
destinations. The initial required-device run `3f59030d` fails on copy size one
against WGPU's `COPY_BUFFER_ALIGNMENT` rule. This is correctness evidence;
no transfer-performance claim is made.
