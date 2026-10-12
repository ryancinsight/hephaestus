# ADR 0076: Device-neutral n-D index-selection seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `index_select` (selecting slices along one axis at
positions given by a 1-D index), the next remaining op after `ScatterOps`.

Like gather, there is no leto oracle to delegate to, so the host reference
carries the direct implementation and the shared clauses pin its value
semantics with hand-computed fixtures; the semantic is the
`torch.index_select` contract `coeus_ops::index_select` documents
(`output[i] = input[i with dim := index[i_dim]]`, where `i_dim` is `i`'s
own `dim` coordinate). Unlike gather's same-rank index, the index here is
rank-1 with one source position per output slice — a distinct kernel, not
a broadcast wrapper, because no broadcast machinery exists to reuse and
the board lists the op separately.

## Decision

`IndexSelectOps<D, T>` in `hephaestus-core::domain::index_select`
provides `index_select_into`, taking a strided input, a `dim`, a
rank-1 strided `u32` index, and a caller-owned strided output. Indices
are `u32`, matching `EmbeddingOps`. `validate_index_select_shape`
checks the geometry in a fixed order (dim range, non-dim agreement,
output `dim` extent against the index length). The index rank is
concrete, so no rank-relation bound is needed; the index length counts
selections and may exceed the input's `dim` extent freely. The output
buffer must not alias the input buffer, and the shared clause pins
rejection-before-mutation. Empty indices select vacuously.

Out-of-range indices follow the embedding seam's contract exactly: the
host rejects them while reading, and device backends report them through
a one-`u32` atomic error flag downloaded after dispatch, failing with
`InvalidConfiguration` and an out-of-range message. No backend clamps
or wraps; the faulting lane's copy is skipped.

`WgpuIndexSelectOps` dispatches one thread per output cell in a single
launch: a thread decodes its output coordinate, reads the index at its
own `dim` coordinate through the rank-1 stride, substitutes it into the
`dim` lane, and copies the addressed input cell. Each thread writes only
its own cell, so the kernel is race-free. When the selection dimension
itself is empty but the output is not, every index is trivially out of
range and the host side fails before dispatch (the kernel could not
even bind its empty source), mirroring the gather seam. The host
reference walks the output odometer and reads through
`leto::ArrayView` directly.

CUDA, ROCm, and Metal do not implement `IndexSelectOps` yet, and the
seam is not folded into `assert_backend_contract`/`BackendUnderTest`
this increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `index_select_into` on WGPU (and the host reference pair)
today. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further: index_put,
masked_fill, tile, where, sort, and nonzero remain. CUDA/ROCm/Metal
implementations and the aggregate fold are follow-up items.
