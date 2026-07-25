# ADR 0018 (hephaestus): prepared map-reduction parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU exposes prepared dot and L2-norm plans over fixed strided inputs. The
plans retain mapped-product scratch, the complete reduction tree, scalar
outputs, and the L2 square-root output. CUDA and ROCm expose the same
immediate dot/norm operations and the prepared scalar reduction substrate, but
not the prepared composite surface. Metal delegates immediate linalg through
WGPU but does not expose the prepared map-reduction names.

## Decision

Extract a reusable reduction-plan core from the CUDA and ROCm prepared scalar
reduction modules. The plan owns the reduction tree and outputs and accepts a
device buffer at dispatch, allowing a composite plan to own its mapped-product
scratch without a self-referential borrow. Add backend-owned `PreparedDot` and
`PreparedL2Norm` plans that validate strided input layouts during preparation,
reuse fixed scratch/output allocations, invoke the existing native strided
elementwise kernels for the map phase, and reuse the prepared native
reduction tree. L2 completes with the existing native square-root kernel.

Add Metal-owned wrappers exposing the same constructors and dispatch/output
roles by delegating to WGPU on the Metal-selected device. The backend boundary
therefore stays Metal-owned while the kernel family remains single-source in
the WGPU implementation.

## Alternatives rejected

- Re-run immediate dot/norm and return newly allocated buffers: this would
  omit fixed output identity and prepared scratch semantics.
- Store a prepared reduction borrowing an internally allocated map buffer:
  Rust self-referential storage is invalid; a buffer-independent reduction
  plan preserves ownership without unsafe indirection.
- Add separate CUDA and ROCm composite kernels: existing native strided map
  and reduction kernels already supply the required computation, so a second
  algorithm would duplicate numerical behavior.
- Add prepared L1, max, or trace in this increment: WGPU does not expose
  equivalent prepared contracts for those operations.

## Verification

Backend contracts compare repeated prepared dot/L2 dispatch with the Leto CPU
reference after rewriting input buffers, assert stable output allocation
identity, cover empty and non-contiguous layouts, and reject mismatched dot
shapes and invalid storage layouts before launch. CUDA, ROCm, and Metal CI
will run focused feature, warning-denied Clippy, nextest, doctest, and rustdoc
gates; hardware lanes remain required-device checks when a self-hosted GPU
label is available.
