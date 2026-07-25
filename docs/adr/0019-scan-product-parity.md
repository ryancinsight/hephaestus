# ADR 0019 (hephaestus): cumulative-product scan parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

The four backends already share the rank-2 strided scan planner and operation
vocabulary. WGPU and CUDA can instantiate the existing generic `CumProdOp`
scan kernels, and ROCm already exposes `cumprod` and `cumprod_into`, but WGPU,
CUDA, and Metal expose only the generic `scan_axis` route. This leaves a
public capability mismatch even though the product computation is already
implemented and tested in the provider paths.

## Decision

Expose `cumprod` and `cumprod_into` on WGPU and CUDA as thin monomorphized
entry points over their existing `CumProdOp` scan kernels. Preserve the
established ROCm convention: cumulative product is a reverse scan, so each
logical output contains the product from its position through the end of the
selected axis. Add Metal-owned wrappers that delegate the same two operations
through its native Metal-selected WGPU device.

The APIs retain the existing `StridedOperand`, `BlockWidth`, layout validation,
output-alias rejection, empty-layout identity, and C-contiguous allocating
output contracts. No product-specific kernel or alternate forward-product
semantics is added.

## Alternatives rejected

- Keep product available only through `scan_axis`: the generic route computes
  the values but leaves the public backend capability matrix asymmetric.
- Duplicate product kernels in each backend: the existing generic scan
  implementation already parameterizes the combine expression and identity,
  so duplication would create divergent numerical and validation behavior.
- Add a forward-product API with a new name: the current ROCm contract uses
  reverse cumulative product, and a second public semantic would not close the
  existing parity gap.

## Verification

WGPU, CUDA, ROCm, and Metal contracts compare both allocating and
caller-owned outputs over both axes and a transposed strided input against
independent integer product references. They also cover zero-sized layouts and
invalid storage layouts. At implementation head `f4c74c3`, the CUDA workflow
passed as run `30177430115` / job `89728432211` in 7m24s, the ROCm workflow
passed as run `30177430114` / job `89728432299` in 6m04s, and the Metal
workflow passed as run `30177430125` / job `89728432239` in 6m50s. Required-
device lanes were skipped because hosted GPU labels were unavailable; they
remain the hardware evidence tier when registered GPU runners exist.
