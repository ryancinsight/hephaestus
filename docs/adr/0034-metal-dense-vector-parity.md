# ADR 0034: Metal dense-vector parity

- Status: accepted
- Date: 2026-07-27
- Scope: `DenseVectorOps` in the Metal provider
- Change class: `[arch]` and `[minor]`

## Context

The shared `DenseVectorOps` seam now has WGPU, CUDA, and ROCm
implementations. The Metal provider already selects the native Metal API
through its WGPU device wrapper, but it did not expose the same typed vector
bundle. Iterative consumers therefore still had to special-case Metal even
though its buffer and dispatch substrate already matched the WGPU contract.

## Decision

Add `MetalVectorOps` as the Metal provider's implementation of the shared
seam. It prepares one `WgpuVectorOps` bundle from the Metal-selected device
and forwards the operation boundary through that already-native dispatch
context. The Metal prepared handles contain only the existing WGPU prepared
resources; they do not clone or retain additional Metal buffers. The GAT
prepared-handle contract continues to bind execution to the allocations used
during preparation and rejects a different allocation at dispatch.

The contract covers device copy, scale, AXPY, XPAY, subtraction, prepared dot,
and prepared L2 norm. Empty vectors remain no-ops, and length 257 exercises
the 256-thread workgroup tail.

## Alternatives rejected

- Add Metal-specific vector kernels: rejected because Metal already executes
  the canonical WGPU kernels through its selected native device and a second
  kernel family would duplicate the operation contract.
- Rebuild reductions for every call: rejected because it would discard the
  prepared-resource reuse guarantee shared by the other backends.
- Retain a separate Metal-only consumer API: rejected because it would leave
  the canonical `DenseVectorOps` seam incomplete.

## Verification

The Metal contract compares all seven operations and empty-vector behavior
with CPU f32 formulas, checks prepared dot reuse after a device-to-device
update, and rejects mismatched lengths. Local Windows verification can only
compile and run adapterless paths; macOS Metal CI supplies the device-backed
contract execution.

## Revisit trigger

Revisit if Metal requires a kernel or prepared-resource lifetime that the
WGPU dispatch substrate cannot express without changing the shared seam.
