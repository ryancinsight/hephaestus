# ADR 0039: Own accelerator convolution in Hephaestus

- Status: Accepted
- Date: 2026-07-30
- Board item: `HEPH-CONVOLUTION-PROVIDER-1`
- Cross-repository driver: Coeus ADR 0046
- Change class: `[arch] [minor]`

## Context

Coeus currently owns CUDA and WGPU convolution kernels, CPU fallback paths,
and transposed-convolution host loops. ROCm and Metal expose no convolution
capability. The consumer contract is infallible, so unsupported scalar,
layout, compilation, launch, and device failures either disappear into a CPU
provider transition or cannot be represented.

Leto now owns scalar- and rank-generic regular and transposed convolution
forward and additive-backward contracts. Hephaestus already owns typed device
buffers, strided views, device acquisition, pipeline caches, and fallible
dispatch. The missing boundary is one accelerator convolution seam and its
device-API implementations.

## Decision

`hephaestus-core::domain::convolution` owns one device-neutral
`ConvolutionOps` family over `ComputeDevice`, `StridedView`, scalar type, tensor
rank, and spatial rank. Its operation boundary includes:

- regular forward;
- regular additive backward with independently selected gradients;
- transposed forward; and
- transposed additive backward with independently selected gradients.

The seam uses Leto's `ConvolutionParameters` and
`TransposedConvolutionParameters` as the parameter SSOT. Stable Rust cannot
express `tensor_rank = spatial_rank + 2` in every signature, so the shared
planner validates that relationship once before dispatch. Associated prepared
types retain compiled kernels, metadata, and any backend-owned resources
needed for repeat dispatch. Generic consumers monomorphize the complete
operation; no dynamic dispatch or per-element capability branch is present.

Shared planning validates rank, shapes, channels, storage spans, arbitrary
writable-layout overlap, buffer aliasing, parameter equations, checked
products, and each backend kernel's address-width contract before compilation
or mutation. Backend metadata construction additionally proves every logical
index product and physical address fits the kernel's integer representation.
Backward preparation compiles every requested gradient kernel before the first
launch. Validation and preparation failures leave caller-owned outputs
unchanged. A device fault surfaced by synchronization or transfer is returned
as a typed error and never changes provider; the contract does not claim
transactional rollback after device execution has begun.

Backend ownership is:

- `hephaestus-wgpu` owns WGSL kernels and prepared command resources;
- `hephaestus-cuda` owns CUDA source/PTX, module loading, streams, and launch
  metadata;
- `hephaestus-rocm` owns HIP source, module loading, streams, and launch
  metadata; and
- `hephaestus-metal` delegates to the WGPU implementation over its existing
  Metal-selected WGPU device and buffer handles, preserving device residency
  without duplicating the mathematical or shader layer.

Each backend admits only scalars with a native kernel implementation.
Unsupported scalar/rank/layout combinations return typed capability errors;
they never download operands, execute Leto, or select another accelerator.

## Alternatives

- Keep convolution kernels in Coeus. Rejected because each consumer would own
  its own vendor dimension and Hephaestus would not be a substrate replacement.
- Add only CUDA and WGPU. Rejected because a generic Coeus adapter would still
  have incomplete ROCm and Metal behavior and latent host defaults.
- Implement separate Metal shaders. Rejected because Metal already uses
  Hephaestus WGPU with `Backends::METAL`; another shader family duplicates the
  same device implementation.
- Permit accelerator-to-CPU fallback. Rejected because reported backend
  identity would differ from the provider executing the operation.

## Consequences

Coeus adapters borrow their existing Hephaestus buffers and convert dynamic
layouts to fixed-rank Leto layouts at one boundary. Coeus-owned CUDA/WGPU
kernels and host fallbacks become obsolete and are deleted after all callers
cut over. No speed, memory, or binary-size improvement is claimed until
matched measurements compare complete pre- and post-cutover operations.

## Verification

- Generic planner tests cover ranks 1 through 3, shape equations, strided
  storage, aliases, empty outputs, checked overflow, and backend address width.
- One generic `hephaestus-conformance` suite runs regular/transposed
  forward/backward against Leto across spatial ranks one through three for
  `f32` and `f64`; each backend instantiates the scalars it supports and
  backend-local tests add address and device-failure cases.
- Exactly representable fixtures use exact equality. Reordered floating-point
  reductions use bounds derived from reduction depth and `f32::EPSILON`.
- Tests select optional bias and every independent gradient target.
- Rejected validation and preparation leave outputs unchanged.
- Static residue scans prove the provider implementations contain no host
  transfer, Leto execution, or provider-transition path.
- Warning-denied package gates, configured Nextest, doctests, SemVer checks,
  and exact-head backend CI pass before consumer cutover.
