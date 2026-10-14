# ADR 0081: Device-neutral n-D tiling seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

- Revised 2026-10-14: the CUDA/ROCm/Metal "not implemented yet" paragraphs below
  are stale 
—
 those backends have since landed (rescue slices 3, 4, 11a). Other
  follow-ups keep their stated status.

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `tile` (repeating the input `reps[d]` times along
every axis), the next remaining op after `RollOps`. It is the
`torch.Tensor.repeat` semantic `coeus_ops::tile` documents, whose
per-element host loop with a division-heavy flat-to-ND conversion per
cell this seam retires for the covered backends.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. The one scoping
decision is rank: where the coeus entry point pads leading size-1 dims
when `reps` is longer or shorter than the rank, this seam takes exactly
one rep per axis (`TileReps<N>`) and keeps input and output at rank `N` —
a rank change stays a reshape-plus-tile composition at the caller, since
every strided seam in the family is fixed-rank and rank-changing views
are the reshape seam's work.

## Decision

`TileOps<D, T>` in `hephaestus-core::domain::tile` provides `tile_into`,
taking a strided input, `reps: TileReps<N>`, and a caller-owned strided
output. `output.shape()[d]` must equal `input.shape()[d] * reps[d]` for
every axis, checked by `validate_tile_shape` in axis order with checked
multiplication: an overflowing rep count is a shape error, never a
wrapped extent. A zero rep count empties its axis legally, and an empty
input stays empty under any reps — either way zero cells copy and no
modulo by zero executes on any backend. The output buffer must not alias
the input buffer, and the shared clause pins rejection-before-mutation.

`WgpuTileOps` dispatches one thread per output cell in a single launch:
decode the coordinate, take each lane modulo the input extent, and copy
the addressed input cell. The rep counts never reach the kernel —
validation proves each output extent is its input extent times its reps,
so the modulo alone reproduces the tiling. Every live lane has a
nonzero input extent (`out = in * reps` with `out > 0` proves `in > 0`),
and padded lanes hold ones, so no modulo by zero executes. The host
reference walks the output odometer through `leto::ArrayView` with the
same modulo, early-outing on any zero output extent.

CUDA, ROCm, and Metal do not implement `TileOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `tile_into` on WGPU (and the host reference pair) today
for the fixed-rank case. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further:
where, sort, and nonzero remain. CUDA/ROCm/Metal implementations and
the aggregate fold are follow-up items.
