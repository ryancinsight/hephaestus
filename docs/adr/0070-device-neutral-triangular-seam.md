# ADR 0070: Device-neutral rank-2 triangular masking seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062,
0069): this record splits out `tril`/`triu` (triangular masking).

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
`rows * cols`): each thread evaluates the same keep/zero test
[`hephaestus_core::triangular_keeps`] provides for the host reference, so
the two backends agree by construction rather than by coincidence. The
signed `diagonal` crosses the uniform boundary as its `u32` bit pattern
(`diagonal as u32` host-side, `bitcast<i32>` in WGSL) since the uniform's
other lanes are already `u32`-typed and WGSL has no signed/unsigned union
type — bitcasting preserves the exact bit pattern, unlike a numeric
narrowing cast, which would clamp or wrap the value instead of round-
tripping it.

CUDA, ROCm, and Metal do not implement `TriangularOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `triangular_into` on WGPU (and the host reference pair)
today. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further: cat, split, stack,
gather, scatter, index_select, index_put, masked_fill, tile, where, sort,
and nonzero remain. CUDA/ROCm/Metal implementations and the aggregate fold
are follow-up items.
