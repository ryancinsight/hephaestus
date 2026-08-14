# ADR 0019 (hephaestus): cumulative-product scan parity

- Status: Accepted
- Class: [minor]
- Date: 2026-07-26

## Context

The four backends already share the rank-2 strided scan planner and operation
vocabulary. Each provider can instantiate the generic `CumProdOp` scan kernel,
but the convenience API must preserve the direction vocabulary used by Leto:
`cumprod` is a forward prefix product and a reverse product is a suffix
operation. The prior provider surface exposed the reverse product under the
name `cumprod`, which diverged from the Coeus CPU contract.

## Decision

Expose `cumprod` and `cumprod_into` on WGPU, CUDA, ROCm, and Metal as thin
monomorphized entry points over the existing `CumProdOp` scan kernels using
`ScanDirection::Forward`. Expose `suffix_prod` and `suffix_prod_into` for the
reverse direction. Metal-owned wrappers delegate both product directions
through its native Metal-selected WGPU device.

The APIs retain the existing `StridedOperand`, `BlockWidth`, layout validation,
output-alias rejection, empty-layout identity, and C-contiguous allocating
output contracts. No product-specific kernel is added.

## Alternatives rejected

- Keep product available only through `scan_axis`: the generic route computes
  the values but leaves the public backend capability matrix asymmetric.
- Duplicate product kernels in each backend: the existing generic scan
  implementation already parameterizes the combine expression and identity,
  so duplication would create divergent numerical and validation behavior.
- Keep reverse semantics under `cumprod`: that conflicts with the Leto
  `ScanDirection` vocabulary and the Coeus CPU `cumprod` contract.

## Verification

WGPU, CUDA, ROCm, and Metal contracts compare both allocating and
caller-owned outputs for forward and reverse product scans over both axes and
a transposed strided input against independent integer product references.
They also cover zero-sized layouts and invalid storage layouts. The exact
merged PR #132 head `7c481b2` passed the CUDA, ROCm, WGPU, and macOS Metal
provider suites (jobs `90190923765`, `90190923670`, `90190923733`, and
`90190923914`). Required-device CUDA and ROCm lanes were skipped because
hosted hardware runners were unavailable; no hardware execution claim is made.
