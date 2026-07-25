# ADR 0016 (hephaestus): prepared axis reduction parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU exposes `PreparedAxisReduction` plans for repeated rank-2 reductions into
fixed output buffers, and the Metal backend delegates that surface through its
native WGPU-Metal device. CUDA and ROCm currently expose only immediate axis
reductions, so repeated callers rebuild the pipeline launch metadata and lose
the prepared-operation contract already available on the other backends.

## Decision

Add `PreparedAxisReduction` plans to CUDA and ROCm. Preparation uses the shared
core axis planner once, caches the backend-native operation kernel, and retains
the validated `AxisReductionMeta` plus fixed input/output layouts for later
dispatch. Sum, min, max, and mean retain their existing operation-specific
identity and empty-axis contracts. Dispatch reuses the retained device
resources and launches on the backend's native stream; batch submission keeps
the prepared plans independent while preserving submission order.

Metal delegates the same public operation names to the existing WGPU prepared
axis implementation on its native Metal-selected device. The plan type remains
backend-owned so buffer and device lifetimes do not cross the backend boundary.

## Alternatives rejected

- Re-run immediate axis reductions from `dispatch`: this rebuilds the prepared
  resources and does not satisfy repeated-dispatch semantics.
- Upload metadata or values to the host on every dispatch: this adds a host
  materialization path to a device-resident operation.
- Add prepared sparse or map-reduction plans in this increment: those have
  distinct storage and multi-input contracts.

## Verification

Backend contracts compare prepared sum, min, max, and mean results with
the Leto CPU reference for both axes and non-contiguous layouts. Repeat and
batch dispatch asserts value semantics; sum empty-axis plans write
their identity, min/max/mean empty-axis plans return their typed
rejection, and invalid axis, width, layout, and alias inputs will be rejected
before launch.
CUDA, ROCm, and Metal CI ran the focused feature, lint, test, doctest, and
rustdoc gates at code head `7728026`: CUDA run `30173211020` / job
`89717642298`, ROCm run `30173211026` / job `89717642411`, and Metal run
`30173211021` / job `89717642467`. Hardware jobs were skipped because hosted
GPU labels were unavailable. WGPU also carries the empty-sum binding
regression test that protects the Metal delegation path.
