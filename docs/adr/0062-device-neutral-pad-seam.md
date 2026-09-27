# ADR 0062: Device-neutral padding seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus accelerator-bridge
  consumer audit (2026-09-26) named `pad` among the shape/indexing family
  operations coeus has no accelerator path for today.

## Context

Coeus's accelerator bridge (`coeus-hephaestus`/`coeus-wgpu`) needs a family of
shape and indexing operations — `cat`, `split`, `stack`, `gather`, `scatter`,
`index_select`, `index_put`, `masked_fill`, `pad`, `roll`, `tile`, `tril`,
`triu`, `where`, `sort`, `nonzero` — none of which Hephaestus exposes as a
device-neutral seam. Per upstream ownership (standards: backend hierarchies),
these land in Hephaestus's generic accelerator layer first, with per-vendor
device implementations, before coeus binds to them.

The family is too large for one item or one trait: each operation has a
distinct index/shape contract, and forcing them into a single mega-trait
would repeat the `assert_backend`-style twenty-parameter struct problem one
op family early. Every existing dense-linalg role in this codebase (ADR 0044)
already splits by contract shape (`DenseProductOps`, `DenseCompositionOps`,
`DenseMatrixFunctionOps`) rather than bundling; the shape/indexing family
follows the same precedent, one seam trait per op (or small op cluster with
an identical contract), landing as separate dependency-free items.

## Decision

`pad` is the family's first delivered seam: `PadOps<D, T>` in
`hephaestus-core::domain::pad`, with one method `pad_into<const N: usize>`
writing a padded copy of a [`StridedView`] into a caller-owned output view.
Its contract mirrors `leto::pad`'s `[(before, after); N]` width array exactly,
so host and device results are directly comparable and the host
implementation (`HostPadOps`) delegates to `leto::pad` rather than
re-deriving the geometry — the host substrate is the correctness reference
(ADR 0046), and reusing the oracle function keeps that reference honest by
construction.

The WGPU implementation (`WgpuPadOps`, `hephaestus-wgpu::application::pad_seam`)
follows the packed rank-8 metadata pattern `hephaestus-wgpu::application::strided`
already established for elementwise dispatch: one kernel serves every rank up
to `MAX_STRIDED_RANK`, decoding the output's flat index into a per-axis
coordinate and mapping it to an input coordinate by subtracting that axis's
`before` width. An input coordinate that underflows past `u32::MAX` (still in
a pad margin) or reaches the input's extent selects the fill value instead of
a source read, so one bounds check per axis covers both margins.

CUDA, ROCm, and Metal do not implement `PadOps` yet. `PadOps` is not folded
into `hephaestus-conformance`'s `assert_backend_contract`/`BackendUnderTest`
aggregate in this increment — that aggregate requires every named seam to
have every backend's implementation, and adding `PadOps` there now would
break CUDA/ROCm/Metal's existing `BackendUnderTest` construction sites for a
seam they do not implement. `assert_pad_contract` stands alone (called
directly from the host and WGPU test suites) until a follow-up item lands
the remaining three implementations and folds the seam into the aggregate.

## Consequences

Coeus can bind `pad` on WGPU (and the host reference pair) today. CUDA/ROCm/
Metal `pad` support, and the remaining fourteen operations of the family
(each its own item per `backlog.md#heph-shape-ops-provider-1`'s decomposition
note), are tracked separately. The rank-8 packed-metadata pattern this ADR
reuses is now proven for a second op family (elementwise, pad); a third
family adopting it strengthens the case for hoisting the encode/decode core
itself into a shared `hephaestus-wgpu` module rather than each seam
re-declaring its own `Meta` struct and WGSL preamble — deferred until a third
consumer makes that consolidation's shared home concrete (consolidation
fires on the second occurrence; the shared home is chosen at the third when
two independent per-seam `Meta` layouts already exist to compare).
