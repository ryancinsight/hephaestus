# Backlog — hephaestus

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## Delivered

- [x] [minor] Add caller-owned contiguous elementwise output APIs
  (`binary_elementwise_into`, `unary_elementwise_into`,
  `scalar_elementwise_into`), route allocating APIs through them, pool scalar
  uniforms, and consolidate WGPU pipeline cache construction. Evidence:
  differential WGPU contract test, fmt, check, clippy, nextest, doctest, docs,
  and empirical `elementwise_into` benchmark on real adapter.
- [x] [patch] Add default `parallel` and `mnemosyne-memory` feature markers to
  `hephaestus-core` and `hephaestus-wgpu`, keeping provider feature policy
  uniform across the Apollo-facing Atlas stack. Evidence: metadata audit, fmt,
  and diff checks; compile/test blocked by Cargo lockfile write/access denial
  before rustc.

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
- [x] [minor] Extend strided dispatch to the unary/scalar op families through
  the same Meta uniform: shared `StridedMeta`/WGSL-decode/`cached_pipeline`/
  `encode_strided` core; scalar family is a zero-new-kernel wrapper over the
  binary kernels (one-element operand, all-zero strides). Verification: unary
  transposed/broadcast and scalar-equivalence tests on real hardware.

## Phase 2: CUDA backend (cuda-oxide + cutile composed) [arch]
- [x] [arch] Gating ADR accepted: `docs/adr/0001-cuda-backend.md` — cuda-oxide
  owns the device substrate (driver/context/streams/memory/transfers, mapping
  one-to-one onto `ComputeDevice`), cutile owns tile/PTX kernel authoring,
  with a strict SoC boundary between them; dynamic driver loading preserves
  no-toolkit-to-compile; adapterless hosts skip like the wgpu suite.
- [ ] [arch] `hephaestus-cuda` stage 1: device substrate on cuda-oxide
  (acquisition, typed `PhantomData<T>` buffers, transfers) + contract tests.
- [ ] [minor] Stage 2: elementwise/reduction kernels via cutile; stage 3:
  strided variants over the shared packed layout metadata.
- [ ] [minor] Differential parity of the CUDA elementwise/reduction dispatch vs
  the wgpu backend and CPU references.

## Phase 2.5: heterogeneous topology integration (atlas ADR 0002) [arch]
- [ ] [minor] Placement-aware allocation: thread themis `PlacementHint` /
  `MemoryTier` (Hbm, Gddr, HostPinned, unified) through `ComputeDevice`
  allocation so consumers select device-memory tiers explicitly; wgpu maps
  hints to buffer usages, CUDA maps to cuMemAlloc/managed/pinned variants.
- [x] [minor] (0.4.0) Topology reporting, wgpu half: `WgpuDevice::topology()`
  populates themis `GpuTopology` from adapter limits/info at acquisition —
  subgroup width + memory tier (integrated→Dram, discrete→Device); wgpu does
  not expose SM/register/shared-mem capacities, so those stay zero per themis
  "never fabricated" law. CUDA half fills the full set from device attributes.
- [x] [minor] (0.5.0) Launch widths from the occupancy pipeline, strided
  family: `BlockWidth` (hephaestus-core, NonZero newtype, DEFAULT 256) flows
  through per-width WGSL generation and a width-keyed pipeline cache
  (`PipelineKey`); operands bundled as `StridedOperand`. Verified on hardware
  at width 128 vs default. Remaining: thread `BlockWidth` through the
  contiguous elementwise/reduction families (they pass their constant into
  the shared key space today).
- [ ] [arch] TPU long-term: `hephaestus-tpu` over the PJRT C API (dynamic
  load, no SDK to compile), only when a consumer drives it; the
  `ComputeDevice` seam already accommodates it. No speculative scaffolding.

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
