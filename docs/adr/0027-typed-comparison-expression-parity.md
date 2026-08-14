# ADR 0027: Typed comparison expression parity

- Status: Accepted
- Date: 2026-07-26
- Scope: Hephaestus elementwise expression vocabulary and WGPU, CUDA, ROCm,
  and Metal binary providers
- Change class: `[minor]`

## Context

Coeus and WGPU already define six value-producing comparisons: equality,
inequality, less-than, greater-than, less-than-or-equal, and
greater-than-or-equal. The existing Hephaestus `BinaryExpr<L>` seam stores one
expression per dialect, which is sufficient for arithmetic but cannot express
comparison masks for multiple scalar representations. WGSL requires typed
`select` literals, while CUDA C++ and HIP C++ require typed conditional-result
literals.

Without a scalar-aware seam, ROCm and Metal provider dispatch had to reject
comparisons even though their Leto CPU contracts and sibling WGPU kernels
already supported them.

## Decision

Add `TypedBinaryExpr<L, T>` to the shared operation vocabulary and add one
comparison marker per operation. Implement f32, i32, and u32 expressions for
WGSL, CUDA C++, and HIP C++. Reuse the existing validation, pipeline caching,
layout decoding, and launch paths by passing the selected expression into
shared private helpers. Export the same markers and typed binary/strided entry
points from all four backend roots; Metal delegates the WGSL implementation
through its native WGPU-Metal device.

The comparison result remains the input scalar type, with values zero or one,
matching Coeus and Leto. f64 and vector comparison markers are not added until
the consumer contracts require those result representations; the typed trait
rejects unsupported instantiations at compile time.

## Alternatives rejected

- Add f32-only comparison markers: rejected because it would leave the shipped
  integer scalar contracts without a native path and would not establish Leto
  parity for all provider scalar implementations.
- Encode comparison literals in `BinaryExpr<L>`: rejected because one
  associated expression cannot vary with `T` without generating invalid shader
  source for at least one scalar family.
- Copy comparison kernels into each Coeus backend: rejected because it would
  duplicate traversal, validation, and dialect expression ownership.
- Fall back to CPU or the direct WGPU kernel for ROCm/Metal: rejected because
  it would hide the missing provider capability instead of closing it.

## Verification

Hephaestus core tests pin representative f32, i32, and u32 expressions for all
three dialects. Coeus ROCm and Metal tests compare all six f32, i32, and u32
operations with the Leto CPU oracle, including broadcasted f32 inputs. The
backend-parity workflow is the authoritative hosted check for WGPU, CUDA,
ROCm, and Metal.

## Revisit trigger

Extend the typed comparison vocabulary when a consumer adds a supported f64,
half-precision, or vector comparison result contract, with an independent
shader-literal and Leto differential test for that representation.
