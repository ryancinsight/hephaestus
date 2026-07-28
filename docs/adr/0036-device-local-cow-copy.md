# ADR 0036: Keep accelerator COW copies on-device

- Status: accepted
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
