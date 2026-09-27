# ADR 0064: Device-neutral batched cross-product seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-device-dot-cross-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

The consumer audit named "device-side dot/cross reductions" as one gap,
observing that coeus copies full operands to host for these. On inspection,
`hephaestus_core::DenseVectorOps::dot` (and `norm_l2`/`norm_l1`/`norm_max`)
already dispatch entirely on-device — `hephaestus-wgpu`'s
`dot_prepared` reduces on the GPU and downloads only the resulting scalar,
never the operands. `dot` was not the actual gap; if coeus is round-tripping
operands for it today, that is a coeus-side integration gap (not binding an
existing hephaestus seam), out of this repository's scope to fix.

Cross product has no seam at all, in any form. Unlike `dot`/`norm_*`, its
output is itself a vector (three components per triple), not a scalar
reduction, so it does not fit `DenseVectorOps`'s reduction-shaped contract —
a new, narrow seam is the right size for it.

## Decision

`CrossProductOps<D, T>` in `hephaestus-core::domain::cross_product` covers
exactly the gap: `cross_into(&self, device, a, b, out)` writes the batched
cross product of two flat arrays of `(x, y, z)` triples. Buffers are flat and
contiguous — `&D::Buffer<T>`, not `StridedView` — matching `DenseVectorOps`'s
own convention rather than the strided generality `PadOps`/`ArgReduceOps`
need: a cross product is defined triple-wise over consecutive elements, and a
caller holding a non-contiguous batch already has to materialize it
contiguously to call `dot`/`norm_l2` today, so this seam asks nothing new of
it.

`HostCrossProductOps` is a direct per-triple loop over the standard formula
(`out.x = a.y*b.z - a.z*b.y`, etc.) — no leto delegation, since leto has no
cross-product function to delegate to and the formula has no numerically
interesting structure worth a library dependency for three multiply-subtract
pairs.

`WgpuCrossProductOps` follows the same one-thread-per-unit, packed-uniform
pattern as `PadOps`/`ArgReduceOps` (ADRs 0062/0063), simplified for the flat
contiguous case: the uniform carries only a triple count, still packed as a
full `vec4<u32>` even for one value — ADR 0063's alignment postmortem is
followed here from the start rather than rediscovered.

CUDA, ROCm, and Metal do not implement `CrossProductOps` yet, and — as with
`PadOps`/`ArgReduceOps` — the seam is not folded into
`assert_backend_contract`/`BackendUnderTest` this increment.

## Consequences

Coeus can bind `cross_into` on WGPU (and the host reference pair) today.
CUDA/ROCm/Metal implementations and the aggregate fold are follow-up items.
The `dot`/`norm_*` half of the originally reported gap needs no hephaestus
change; closing it is a coeus-side binding task, reported back rather than
duplicated here.
