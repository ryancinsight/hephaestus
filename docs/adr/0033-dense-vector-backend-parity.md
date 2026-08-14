# ADR 0033: Dense-vector backend parity

- Status: Accepted
- Date: 2026-07-27
- Scope: `DenseVectorOps` in Hephaestus core, CUDA, and ROCm
- Change class: `[arch]` and `[minor]`

## Context

`DenseVectorOps` defines the vector recurrence and prepared-reduction contract
used by iterative consumers. WGPU implemented the contract, but CUDA and ROCm
did not, leaving the provider seam incomplete. CUDA and ROCm device buffers are
owning RAII values and are intentionally not cloneable, so a prepared handle
must borrow the allocations it is bound to. The previous associated-type shape
could not express that relationship, while the existing strided descriptor also
coupled a buffer lifetime to temporary layout metadata.

## Decision

Use generic associated types for `PreparedDot` and `PreparedNorm`, with the
operand lifetime carried by each prepared handle. CUDA and ROCm implement the
complete f32 contract: device copy, scale, AXPY, XPAY, subtraction, prepared
dot, and prepared L2 norm.

Prepared CUDA and ROCm reduction plans own a cheap clone of the device handle
and borrow only the device buffers. Layout validation and packed metadata are
completed during preparation through an internal buffer/layout-separated path.
This allows dense-vector preparation to use stack-local contiguous layouts
without retaining temporary metadata, and it avoids cloning RAII buffers,
raw-pointer ownership, or allocation during repeated dispatch.

## Alternatives rejected

- Keep non-generic prepared associated types: rejected because they cannot
  represent non-cloneable borrowed CUDA/ROCm buffers safely.
- Clone or reference-count device buffers: rejected because it changes the
  ownership contract and adds lifetime/fragmentation overhead to prepared
  handles.
- Rebuild a reduction plan on every prepared call: rejected because it breaks
  the no-repeated-allocation prepared-operation contract.
- Add backend-local adapters: rejected because the shared trait is the
  canonical consumer-facing seam and adapters would duplicate its contract.

## Verification

Core and provider integration targets compile with no default features. CUDA
and ROCm contract tests exercise all vector operations against f32 CPU
references at length 257, which crosses the 256-thread workgroup boundary;
they also verify prepared-handle reuse after device-to-device input updates
and typed length rejection. WGPU remains covered by its existing dense-vector
suite. Vendor execution is provided by the CUDA and ROCm CI jobs; local
Windows verification cannot claim physical GPU execution.

## Revisit trigger

Revisit when Metal receives the same vector seam, when a non-f32 vector scalar
contract is required, or when a backend requires a prepared resource lifetime
that cannot be represented by the current operand-lifetime GAT.
