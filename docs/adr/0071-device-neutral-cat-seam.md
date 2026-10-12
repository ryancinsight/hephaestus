# ADR 0071: Device-neutral n-D concatenation seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `cat` (concatenation along an existing axis), the
first remaining op after `PadOps` and `TriangularOps`.

Unlike every other seam in this family, `cat` takes a runtime-count input
list: N strided views sharing every dimension except the concatenation
axis. Coeus calls it pervasively (`coeus_ops::cat` backs autograd `cat`,
and the RNN cells concatenate one output per timestep), so the seam must
serve both small-N (2-3 inputs) and unrolled large-N calls without an
arity cap.

## Decision

`CatOps<D, T>` in `hephaestus-core::domain::cat` provides `cat_into`,
taking `&[StridedView<D::Buffer<T>, N>]`, an `axis`, and a caller-owned
strided output. `validate_cat_shape` checks the geometry in a fixed order
(axis range, non-empty list, non-axis agreement, checked axis summation,
output shape) and returns the per-input axis bases, so backends never
re-derive placement. The output buffer must not alias any input buffer;
both backends reject aliasing with the same `DispatchFailed` message, and
the shared clause pins rejection-before-mutation. Empty segments are
legal and contribute nothing.

`WgpuCatOps` dispatches one block-copy pass per input inside a single
encoder, all passes sharing one pipeline: a thread decodes its input
coordinate, reads through the input strides, and writes the same
coordinate with the `axis` lane shifted by the validated base. No bounds
check is needed because validation proves each output window equals its
input shape exactly. WebGPU has no portable multi-buffer binding for a
runtime input count (binding arrays need non-uniform-indexing features no
other seam requires), so per-input passes are the portable strategy; a
fused single-launch kernel stays future work for a profiled need. Each
pass holds its pooled uniform buffer until the shared submit: dropping
one returns it to the recycle pool, which would hand every bind group the
same buffer with the last pass's metadata. The host reference delegates
to `leto::concat` (ADR 0046) and scatters into the strided output.

CUDA, ROCm, and Metal do not implement `CatOps` yet, and the seam is not
folded into `assert_backend_contract`/`BackendUnderTest` this increment,
for the reason ADR 0062 gives for `PadOps`. Those backends can dispatch
single-launch with packed pointer arrays (no binding-model obstacle
there); the WGPU per-input strategy is a backend-local choice the seam
permits, not a contract other backends must copy.

## Consequences

Coeus can bind `cat_into` on WGPU (and the host reference pair) today.
`HEPH-SHAPE-OPS-PROVIDER-1` narrows further: split, stack, gather,
scatter, index_select, index_put, masked_fill, tile, where, sort, and
nonzero remain. CUDA/ROCm/Metal implementations and the aggregate fold
are follow-up items.
