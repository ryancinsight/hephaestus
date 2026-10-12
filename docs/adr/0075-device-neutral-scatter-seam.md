# ADR 0075: Device-neutral n-D scatter-add seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `scatter` (indexed accumulation along one axis),
the next remaining op after `GatherOps`.

Coeus consumes only `scatter_add` (the backward operator of gather), so
the seam is scatter-add, not overwrite-scatter: the output starts as a
copy of the input, then each source cell accumulates into the cell its
index names. Like gather, there is no leto oracle to delegate to, so the
host reference carries the direct implementation and the shared clauses
pin its value semantics with hand-computed fixtures. Duplicate indices
make input-parallel accumulation nondeterministic (floating-point sums
are order-sensitive, and WGSL has no floating-point atomics), so
accumulation order is part of the contract: ascending row-major source
order, observably pinned by an `f32` fixture whose sum differs under any
other order.

## Decision

`ScatterOps<D, T>` in `hephaestus-core::domain::scatter` provides
`scatter_add_into`, taking a strided input, a `dim`, a same-rank strided
`u32` index, a strided source, and a caller-owned strided output.
`validate_scatter_shape` checks the geometry in a fixed order (dim
range, index/source agreement, non-dim index extents inside the input's,
output-equals-input shape). Only non-`dim` extents must fit: the `dim`
extent counts updates, and repeated indices accumulate in order, so it
may exceed the input's `dim` extent freely. The output buffer must not
alias the input or source buffer (the `u32` index buffer cannot name the
output's `T` buffer, so no third check exists), and the shared clause
pins rejection-before-mutation for both pairs. Empty indices scatter
vacuously, reproducing the input.

Out-of-range indices follow the embedding seam's contract exactly: the
host rejects them while reading, and device backends report them through
a one-`u32` atomic error flag downloaded after dispatch, failing with
`InvalidConfiguration` and an out-of-range message. No backend clamps
or wraps; a faulting update maps to no cell and is skipped.

`WgpuScatterOps` scans output-parallel, one thread per output cell over
every source cell in ascending row-major order — the deterministic form
the embedding scatter-add establishes. A source cell maps to its
thread's cell exactly when the output offset recomputed from its
coordinates equals the thread's own offset, which is sound only because
validation proves the output layout injective. The copy and the scan
are two launches, not one: the scan's four storage buffers (indices,
source, output, error flag) already fill the portable
`max_storage_buffers_per_shader_stage` limit of four, so the input
cannot join the same bind group. The copy kernel runs first (and alone
when the index is empty, when the empty index/source buffers cannot be
bound at all); the scan seeds from the copied output cell, and each
thread still touches only its own cell. `+` applies to every WGSL
dialect scalar, so the kernel needs no further scalar bound.

CUDA, ROCm, and Metal do not implement `ScatterOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `scatter_add_into` on WGPU (and the host reference pair)
today. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further: index_select,
index_put, masked_fill, tile, where, sort, and nonzero remain.
CUDA/ROCm/Metal implementations and the aggregate fold are follow-up
items.
