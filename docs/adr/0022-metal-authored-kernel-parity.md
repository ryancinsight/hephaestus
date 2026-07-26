# ADR 0022 (hephaestus): Metal authored-kernel parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU, CUDA, and ROCm expose the backend-neutral authored-kernel stream seams
and storage-kernel dispatch contracts. Metal already selects WGPU's native
Metal adapter for its operator families, but did not expose those provider
surfaces at its own crate boundary. This left Metal unable to substitute for
the other backends in consumers that author kernels or use multi-storage
dispatch.

## Decision

Add Metal-owned wrappers implementing `KernelDevice` and
`GroupedKernelDevice`. The wrappers adapt Metal buffers to the underlying
WGPU/Metal stream while preserving the core binding, ordering, copy, fill, and
grouped-sequence contracts. Add Metal-owned storage binding and kernel wrapper
types for multi-storage, unary, and binary dispatch; these wrappers delegate
to the existing WGPU storage kernels on the Metal-selected device.

The Metal crate owns the public wrapper types and exports. The WGPU crate
remains the implementation owner for the shared shader path; no consumer sees
WGPU buffer types through a Metal API.

## Alternatives rejected

- Re-export WGPU stream and storage types: their `KernelDevice` and buffer
  implementations are bound to `WgpuDevice`, so re-exports would not satisfy
  the Metal provider contract.
- Duplicate WGSL kernels or add a second `metal-rs` implementation: Metal
  already has a real native Metal path through WGPU, and duplication would
  fork the implementation family.
- Add a dynamic backend adapter: the core seams are statically dispatched and
  the backend set is closed at each provider boundary.

## Verification

Metal contract tests compare authored-kernel and storage-kernel outputs with
explicit value assertions, including grouped ordering and invalid-length
rejection. Implementation head `d200879` passed the CUDA feature and
adapterless lane (run `30180682356`, job `89736582959`, 7m39s), ROCm feature
and adapterless lane (run `30180682371`, job `89736559954`, 5m52s), and macOS
Metal lane (run `30180682365`, job `89736559801`, 5m17s). NVIDIA hardware job
`89736583231` and AMD hardware job `89736560290` were skipped because hosted
hardware labels were unavailable; those skips remain an explicit hardware
evidence limitation.
