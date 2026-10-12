# ADR 0082: Device-neutral n-D stable-sort seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26); ADR 0043 (IEEE special
  values per dialect).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `sort` (stably sorting a strided input along one
axis into the sorted values plus the argsort positions), the next
remaining op after `WhereOps`. It is the `torch.sort` semantic
`coeus_ops::sort` documents — stable, per-axis, with descending — except
the indices are `u32` rather than `T`-encoded: coeus converts positions
into its scalar because its kernels predate the seam, but the seam's
index width is a seam-level choice like every other index in the family,
never a caller decision.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. Sorting interprets `T`
(an ordering with non-finite edges) rather than copying it, so the seam
carries `type Dialect: KernelDialect` like `MaskedFillOps` and
`WhereOps` — but the ADR 0043 const branch skips wholesale instead of
per cell: one unordered `NaN` comparison can corrupt a whole lane's
order on a non-advertising dialect, so no assertion about a `NaN`-mixed
lane can hold on WGSL, while `NaN`-free finite edges (`-0.0` stable
with `+0.0`, `-inf` below everything finite) pin on every backend.

## Decision

`SortOps<D, T: Pod + PartialOrd>` in `hephaestus-core::domain::sort`
provides `sort_into`, taking a strided input, `dim`, `descending`, and
two caller-owned strided outputs — the sorted values and the `u32`
argsort indices. The ordering bound is inline `PartialOrd`, the
`TopKOps` precedent, not a new trait. Input, values, and indices share
rank `N` and must share shape exactly — a sort permutes cells along one
axis, never resizes — checked by `validate_sort_shape` in a fixed order
(`dim` range, values shape, indices shape). The values buffer must not
alias the input buffer, pinned rejection-before-mutation; the indices
buffer is `u32` against the input's `T` and can never name the same
storage. Empty shapes sort vacuously.

Stability is structural: backends order by `(value, original position)`,
so equal values keep their input order in both directions, observably
through the indices. `NaN` sorts as greater than every other value —
last ascending, first descending, stable among NaNs — on
IEEE-advertising dialects only.

`WgpuSortOps` dispatches one thread per axis lane in a single launch:
decode the lane's base offsets with the padded rank-8 odometer (the
sort axis passes as the padded `dim_lane`, `8 - N + dim`, the
gather/index-select convention), then insertion-sort the lane streaming
from the input, maintaining `(value, position)` pairs directly in the
output slots the way the top-k seam does. The bubble step swaps only on
a strict comparison, which makes the in-shader sort stable by
construction. Three storage buffers plus the uniform stay within the
portable limit. The host reference gathers each lane through
`leto::ArrayView` and sorts `(position, value)` pairs with a
`NaN`-greatest comparator in Rust's IEEE arithmetic, keeping the
position tiebreak ascending in both directions.

CUDA, ROCm, and Metal do not implement `SortOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `sort_into` on WGPU (and the host reference pair) today,
converting the `u32` positions into its `T`-encoded indices at the
bridge. `HEPH-SHAPE-OPS-PROVIDER-1` narrows to one op: only nonzero
remains. CUDA/ROCm/Metal implementations and the aggregate fold are
follow-up items; the CUDA and ROCm residuals will exercise the
`NaN`-greatest cells their dialects advertise.
