# ADR 0011: Aequitas stencil spacing

- Status: Accepted
- Date: 2026-07-19
- Class: [arch]

## Context

Provider-owned finite-difference kernels require physical grid spacing while
their device parameter blocks contain scalar inverse-square coefficients.
Accepting unlabelled `f32` spacing in the public operation contract permits
centimetres, millimetres, and metres to be mixed before the dispatch boundary.

## Decision

- `Laplacian2DParams::new` accepts Aequitas `Length<f32>` values and an
  explicit `LaplacianPolarity`.
- Leto owns the generic `Laplacian2D<T>`, boundary, and polarity contracts.
  The Hephaestus constructor delegates validation and canonical-metre
  conversion to that contract, then stores its signed inverse-square
  coefficients in the POD parameter block.
- WGSL and buffer storage remain monomorphic `f32`; Aequitas does not enter the
  device ABI or the per-element loop.
- Consumers construct typed lengths and select the operation polarity before
  calling the provider; they do not duplicate stencil validation.

## Alternatives rejected

- Keep raw scalars and document metres: rejected because documentation cannot
  prevent unit mismatch.
- Store quantities in the WGSL parameter block: rejected because type-level
  dimensions are a host contract and must not alter the device ABI.
- Make Hephaestus own Cartesian grid types: rejected because CPU and accelerator
  providers require the same contract, whose deepest common owner is Leto.

## Consequences

The new stencil surface is dimensionally checked from its first published
version. Unit conversion occurs once at construction and compiles away from
dispatch and kernel execution.
