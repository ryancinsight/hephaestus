# ADR 0040: Own accelerator attention in Hephaestus

- Status: Accepted
- Date: 2026-07-31
- Board item: `HEPH-ATTENTION-PROVIDER-1`
- Cross-repository driver: Coeus backend dispatch
- Change class: `[arch] [minor]`

## Context

Coeus currently owns separate CUDA and WGPU scaled dot-product attention
kernels plus host fallback paths. Its CPU implementation also owns the same
mathematics independently. ROCm and Metal expose no attention capability, and
infallible consumer entry points cannot surface unsupported layouts, kernel
preparation failures, or device faults.

Leto owns the borrowed rank-3 CPU reference contract for attention forward and
additive backward, including arbitrary strides, broadcast keep masks, causal
masking, fully masked rows, finite arithmetic, and failure-atomic validation.
Hephaestus owns accelerator buffers, strided views, pipeline preparation, and
fallible device dispatch. The missing boundary is one accelerator attention
seam implemented by every device provider.

## Decision

`hephaestus-core::domain::attention` owns rank-3 borrowed operands, masking,
selected additive-gradient destinations, validation, planning, and one
`AttentionOps<D, T>` dispatch seam over `D: ComputeDevice`. Associated prepared
types retain compiled kernels, metadata, and backend resources for repeated
forward and backward dispatch. Generic consumers monomorphize the operation at
the backend and scalar boundary; the hot path contains no vtable or per-element
provider branch.

Query, key, value, output, weights, upstream gradients, and selected gradient
destinations remain device-resident. Callers own all public output storage.
Keep masks borrow a rank-2 `[mask_batch, key_sequence]` device view and carry a
nonzero execution-batch group width. This represents rank-1, singleton-batch,
input-batch, and execution-batch masks without materialization. The scale
remains a runtime scalar because sequence feature width is data-dependent.

Validation proves shape equations, storage spans, writable-layout uniqueness,
buffer non-aliasing, finite scale, checked products, and backend address width
before compilation or mutation. Backward preparation compiles every selected
gradient kernel before the first launch. Validation and preparation failures
leave destinations unchanged. Once dispatch starts, device faults are returned
without changing provider; transactional rollback is not claimed.

Backend ownership is:

- `hephaestus-wgpu` owns WGSL attention kernels and prepared resources;
- `hephaestus-cuda` owns CUDA kernels, module loading, and launch metadata;
- `hephaestus-rocm` owns HIP kernels, module loading, and launch metadata; and
- `hephaestus-metal` delegates to the WGPU implementation over its existing
  Metal-selected WGPU device, preserving residency without duplicating shaders.

Each provider admits only scalar and layout combinations implemented natively.
Unsupported capability returns a typed error and never downloads operands,
executes Leto, or selects another backend.

Coeus dispatches CPU tensors directly to Leto and accelerator tensors directly
to the matching Hephaestus implementation. Its CUDA/WGPU attention kernels and
host fallbacks are deleted in the same consumer cutover.

## Alternatives

- Keep attention in Coeus. Rejected because the consumer would continue to own
  vendor kernels and duplicate Leto's CPU contract.
- Add only CUDA and WGPU. Rejected because ROCm and Metal would retain an
  incomplete backend contract or require hidden fallback.
- Use dynamic dispatch. Rejected because the backend set is closed at the
  consumer boundary and enum/generic routing preserves static kernel dispatch.
- Permit accelerator-to-CPU fallback. Rejected because reported backend
  identity would differ from the provider that executed the operation.

## Consequences

The provider boundary is fallible and therefore requires a Coeus attention
error-surface migration. Device storage remains borrowed and no public
operation allocates host tensor copies. No runtime, memory, or artifact-size
improvement is claimed until matched measurements compare complete operations.

## Verification

- Shared planner tests cover shape equations, arbitrary input strides,
  broadcast masks, output uniqueness, aliases, empty axes, finite scale, and
  checked overflow.
- One generic conformance suite compares each available provider with Leto for
  unmasked, causal, keep-mask, causal-plus-keep, and fully masked forward cases
  plus every independently selected backward destination.
- Exactly ordered fixtures use exact equality. Reordered reductions use bounds
  derived from reduction depth and scalar machine epsilon.
- Rejected validation and preparation preserve caller-owned destinations.
- Residue scans prove provider implementations contain no host execution,
  download, or provider-transition path and that Coeus retains no local
  attention kernel.
- Warning-denied package gates, configured Nextest, doctests, SemVer checks,
  independent review, and exact-head backend CI pass before merge.
