# Backlog — hephaestus

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## Phase 1: wgpu substrate (0.1.0) [arch]
- [x] [arch] Scaffold workspace: `hephaestus-core` contracts (ComputeDevice GAT
  seam, DeviceBuffer, error vocabulary) + `hephaestus-wgpu` backend (acquisition,
  typed buffers, upload/download, elementwise ZST-op dispatch). Differential
  contract tests pass on real hardware; fmt/clippy/test/doc gates clean.
- [x] [minor] Pipeline + shader-module caching keyed by `(Op, T)` so repeated
  dispatch skips recompilation (mirrors apollo's per-kernel caches).
- [x] [minor] Unary elementwise dispatch (ZST markers, shared WGSL template) and
  scalar-broadcast variants, mirroring leto-ops' op families on-device.
- [x] [minor] Reduction dispatch (sum/min/max) with workgroup-tree reduction.
- [x] [minor] Strided-layout-aware dispatch reusing leto host-side `Layout<N>`
  metadata: `binary_elementwise_strided_into` (rank ≤ 4, compile-time capped)
  broadcasts inputs to the output shape with leto rules, writes through a
  caller-owned output buffer, rejects zero-stride-aliasing outputs, and packs
  shape/strides/offsets in one 80-byte uniform. Verification: differential
  tests vs CPU references over identical layouts (transposed, dual-broadcast,
  offset sub-block, rank-3 inner-transpose, rejections) on real hardware.
- [ ] [minor] Extend strided dispatch to the unary/scalar op families through
  the same Meta uniform (shared template; no traversal duplication).

## Phase 2: CUDA backend (cuda-oxide + cutile composed) [arch]
- [ ] [arch] `hephaestus-cuda`: ComputeDevice impl with cuda-oxide owning
  driver/runtime/device-memory/streams and cutile owning tile/PTX kernel
  authoring. Dynamic driver loading; compiles without a CUDA toolkit. ADR in
  this repo before implementation.
- [ ] [minor] Differential parity of the CUDA elementwise/reduction dispatch vs
  the wgpu backend and CPU references.

## Phase 3: memory + ownership integration [minor]
- [ ] [minor] Consume mnemosyne device pools / pinned-host staging (mnemosyne
  Stage D1) for buffer allocation instead of direct device allocation.
- [ ] [minor] melinoe-branded device buffers: ownership transfer across
  host/device/stream as compile-time proofs (melinoe Stage D1 pattern).

## Phase 4: consumers [arch]
- [x] [minor] apollo: `apollo-wgpu-helpers` delegates acquisition to
  `hephaestus-wgpu` with its public API preserved.
- [ ] [arch] coeus: re-base `coeus-wgpu`/`coeus-cuda` onto hephaestus once coeus
  bumps wgpu 23 → 26 (coeus MS-60+ Stage D).
- [ ] [minor] moirai: GPU co-scheduling adapter over hephaestus (moirai Stage D).
