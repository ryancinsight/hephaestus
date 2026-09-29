# ADR 0070: Device-neutral rank-2 triangular masking seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `tril`/`triu` (triangular masking).

Unlike every other seam in this family, triangular masking has no `axis`
parameter — the kept/zeroed region is a function of each element's
`(row, col)` position relative to a diagonal, evaluated over the whole 2D
view at once, not by scanning one axis at fixed positions on the other.

## Decision

`TriangularOps<D, T>` in `hephaestus-core::domain::triangular` provides
`triangular_into`, taking a `TriangularMode` (`Lower` | `Upper`, matching
numpy's `tril`/`triu` naming) and a signed `diagonal` offset. Element
`(row, col)` is kept when `col <= row + diagonal` (`Lower`) or
`col >= row + diagonal` (`Upper`), and zeroed otherwise — `diagonal = 0` is
the main diagonal; positive shifts the kept region up-right, negative
down-left, matching numpy's convention exactly (pinned per
`numerical_discipline`: convention pinning) so a coeus caller porting numpy
code needs no sign translation.

`WgpuTriangularOps` dispatches one thread per `(row, col)` pair (flat over
`rows * cols`). Its zero-copy storage contract is limited to scalar WGSL
storage types whose Rust element stride matches the WGSL array stride; vector
arrays such as Rust `[f32; 3]` are rejected because WGSL `vec3` array elements
have padding that Rust arrays do not. Both layouts must fit their buffers, the
output layout must be injective, and input/output buffers must not alias.
Element offsets and strides are signed `i32` shader addresses, and dispatch
workgroups are checked against the acquired device's
`max_compute_workgroups_per_dimension` before submission. The core API and
WGPU accept the full `i64` diagonal range. WGPU clamps the threshold to `i32`
before uniform upload: validated row and column counts keep each coordinate in
`[0, i32::MAX - 1]`, so `col - row` lies in `[-(i32::MAX - 1), i32::MAX - 1]`.
Clamping the threshold to the `i32` interval preserves both `<=` and `>=`
comparisons against that coordinate-difference interval. Empty views return
before dispatch and also accept any `i64` diagonal. The host reference uses
[`hephaestus_core::triangular_keeps`], and the shared contract checks both
backends against the same full-range value oracle. The clamped `i32`
threshold crosses the uniform boundary as its `u32` bit pattern
(`diagonal_i32 as u32` host-side, `bitcast<i32>` in WGSL) because the
uniform's other lanes are `u32`-typed and WGSL has no signed/unsigned union
type.

CUDA, ROCm, and Metal do not implement `TriangularOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `triangular_into` on WGPU (and the host reference pair)
today. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further: cat, split, stack,
gather, scatter, index_select, index_put, masked_fill, tile, where, sort,
and nonzero remain. CUDA/ROCm/Metal implementations and the aggregate fold
are follow-up items.
