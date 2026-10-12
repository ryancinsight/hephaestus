# ADR 0074: Device-neutral n-D gather seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `gather` (index-based element selection along one
axis), the next remaining op after `StackOps`.

Gather differs from the copy-shape ops before it in two ways. First, there
is no `leto::gather` to delegate to, so the host reference carries the
direct implementation and the shared clauses pin its value semantics with
hand-computed fixtures; the semantic is the `torch.gather` contract
`coeus_ops::gather` documents
(`output[i] = input[i with dim := index[i]]`). Second, placement is
data-dependent: an index value at or past the input's `dim` extent is a
fault no shape check can catch before dispatch.

## Decision

`GatherOps<D, T>` in `hephaestus-core::domain::gather` provides
`gather_into`, taking a strided input, a `dim`, a same-rank strided
`u32` index, and a caller-owned strided output. Indices are `u32`,
matching `EmbeddingOps`: the index width is a seam-level choice, not a
caller decision. `validate_gather_shape` checks the geometry in a fixed
order (dim range, input/index non-dim agreement, output-equals-index
shape) and returns nothing: placement comes from the index data.
The output buffer must not alias the input buffer; the index buffer
holds `u32` against the output's `T`, so the two cannot name one buffer
and no second alias check exists. Empty indices gather vacuously.

Out-of-range indices follow the embedding seam's contract exactly: the
host rejects them while reading, and device backends report them through
a one-`u32` atomic error flag downloaded after dispatch, failing with
`InvalidConfiguration` and an out-of-range message. No backend clamps
or wraps; the faulting lane's copy is skipped, and partial output on
failure is irrelevant once the call returns an error.

`WgpuGatherOps` dispatches one thread per output cell in a single
launch: a thread decodes its output coordinate, reads the index at its
own position, substitutes it into the `dim` lane, and copies the
addressed input cell. When the gather dimension itself is empty but the
index is not, every index is trivially out of range and the host side
fails before dispatch (the kernel could not even bind its empty
source), mirroring the embedding scatter's empty-table early error. The
host reference reads through `leto::ArrayView` directly and scatters
into the strided output.

CUDA, ROCm, and Metal do not implement `GatherOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `gather_into` on WGPU (and the host reference pair) today.
`HEPH-SHAPE-OPS-PROVIDER-1` narrows further: scatter, index_select,
index_put, masked_fill, tile, where, sort, and nonzero remain.
CUDA/ROCm/Metal implementations and the aggregate fold are follow-up
items.
