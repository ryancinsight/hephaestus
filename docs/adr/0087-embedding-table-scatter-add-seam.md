# ADR 0087: Embedding-table scatter-add backward seam

- Status: Accepted
- Refs: `backlog.md#heph-embedding-gather-provider-1` (gradient half,
  deferred by ADR 0065); coeus accelerator-bridge consumer audit
  (2026-09-26); `ScatterOps` (ADR 0077).

## Context

ADR 0065 delivered the embedding-table gather forward pass and named its
gradient counterpart as follow-up: the backward pass for the lookup layer's
training path. The gradient scatters upstream rows into a table-shaped
gradient buffer at the same indices the gather read, accumulating repeats.
Scatter-add is not commutative-associative in float rounding across
execution orders, so the accumulation order is a contract decision, not an
implementation detail. The same error-handling restraint as the gather
applies: indices are device-resident, so range faults surface as typed
errors, never clamped or wrapped reads.

## Decision

`EmbeddingOps<D, T>` gains `scatter_add_into`, accumulating
`grad_table[indices[i]] += grad_output[i]` in place, each table cell summing
its contributors in ascending index-position order. The order pins bitwise
agreement across backends; the conformance suite proves it with an
order-sensitive f32 fixture (`1e20 + 1 - 1e20 + 1 == 1.0` ascending, `0.0`
under any tree or reverse reduction).

`HostEmbeddingOps` runs the ascending loop directly (`T: Pod + Copy +
AddAssign`, the same bound the host `ScatterOps`/`IndexPutOps`
implementors carry). `WgpuEmbeddingOps` runs two passes sharing the
one-`u32` atomic error flag the gather introduced: a range-check pass (one
thread per index) then an output-parallel accumulate pass (one thread per
table row sweeping all `n` positions). The split exists for the
empty-table edge — with zero table rows no accumulate thread runs, so only
the dedicated check can observe a fault — and keeps the accumulate inner
loop free of range branches. An input-parallel scatter would race on
repeated indices, and WGSL has no floating-point atomics, so the
output-parallel scan is what makes the ascending-order contract
executable at all, exactly as ADR 0077 decided for n-D scatter.

## Consequences

Coeus can bind the embedding backward on WGPU (and the host reference
pair) today. CUDA/ROCm implementors and the `assert_backend_contract`
aggregate fold remain follow-ups, as with the gather. The gather uniform
is renamed `GatherMeta` to `EmbeddingMeta` since both passes share the
one `vec4<u32>` layout; no wire or value change.
