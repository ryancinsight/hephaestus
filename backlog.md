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
- [ ] [minor] Strided-layout-aware dispatch reusing leto host-side `Layout<N>`
  metadata (shape/stride uniform buffer) so consumers avoid materializing
  contiguous copies before dispatch.

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
