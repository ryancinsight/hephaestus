# ADR 0035: Dense-vector elementwise parity

- Status: Accepted
- Date: 2026-07-27
- Scope: `DenseVectorOps` elementwise operations in Hephaestus core, WGPU,
  CUDA, ROCm, and Metal
- Change class: `[minor]`

## Context

Leto's `Scalar` contract provides dense-slice addition, subtraction,
multiplication, and division. The shared Hephaestus vector seam already
provided subtraction, but a consumer still had to leave the seam to express
the other binary arithmetic operations. That split prevented one generic
vector recurrence from selecting the same caller-owned storage strategy on
every provider.

## Decision

Extend `DenseVectorOps` with `add_into`, `multiply_into`, and `divide_into`.
Each method validates equal dense lengths, treats an empty input as a no-op,
and writes into distinct caller-owned output storage. Provider implementations
delegate to their existing typed binary-elementwise paths; no new provider
algorithm or result-buffer allocation is introduced. Metal forwards through
its native-Metal-selected WGPU bundle, preserving one canonical operation
implementation for that backend.

The operation set remains f32 for this seam. Integer and reduced-precision
vector contracts require their own numerical and provider capability audit;
they are not inferred from the f32 implementation.

## Alternatives rejected

- Return a newly allocated buffer from each operation: rejected because it
  breaks the allocation-free caller-owned-output contract and increases
  iterative solver memory pressure.
- Add provider-specific `add`, `mul`, and `div` methods: rejected because it
  would fork the consumer seam and make backend parity a naming convention
  rather than a trait guarantee.
- Add a second set of provider kernels: rejected because existing typed
  elementwise paths already provide the required native dispatch.

## Verification

WGPU, CUDA, ROCm, and Metal contracts compare all three new operations against
  native f32 CPU formulas, reuse one output allocation across subtraction,
  addition, multiplication, and division, cover empty inputs and a 257-element
  workgroup tail, and reject mismatched lengths. Provider CI supplies the
  CUDA, ROCm, WGPU, and macOS Metal execution evidence.

## Revisit trigger

Revisit when Coeus consumes the seam for a non-f32 scalar or when a provider's
typed elementwise path cannot preserve the caller-owned-output contract.
