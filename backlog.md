# Backlog — hephaestus

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## Delivered

- [x] [patch] Close the `matrix_rank`/`det` ill-conditioned divergence residuals
  with documentation + testing: documented the relative-threshold (`matrix_rank`)
  and no-determinant-tolerance (`det`) contracts on the public APIs, added
  threshold-boundary and near-singular contract tests
  (`matrix_rank_relative_tolerance_is_the_discriminator`,
  `det_of_near_singular_triangular_is_exact_pivot_product`), and restructured
  `gap_audit.md` into an honest SSOT (Resolved / Accepted design / Open future
  work / Environment). Evidence: analytically-derived value-semantic tests + Leto
  differential; full workspace gate; clippy `-D warnings`.
- [x] [patch] Make WGPU staging-pointer→mapped-block resolution `O(log n)`:
  `WGPU_MAPPED_BUFFERS` is now a base-address-keyed `BTreeMap`; the two
  HostPinned alloc/upload sites share one `resolve_mapped_buffer` helper doing a
  `range(..=ptr).next_back()` containment query instead of an `O(n)` linear scan
  under the global lock. Tightened the registry + descriptor to `pub(crate)`
  (no external consumers) and removed the dead `WgpuMappedBuffer::usage` field.
  Evidence: `test_placement_aware_allocation`, upload/download round-trip, and
  write-buffer contract tests; full 228-test workspace gate; clippy `-D warnings`.
- [x] [minor] Add checked `BlockWidth` grid-count arithmetic in core and route
  WGPU dispatch validation through it, keeping overflow detection in one
  type-level launch-policy API. Evidence: value-semantic launch and WGPU
  workgroup boundary tests, static diagnostics, and full gate.
- [x] [patch] Route scalar uniform, strided metadata uniform, and singleton
  reduction copy sizing through the shared checked WGPU byte-size helper.
  Evidence: byte-size overflow unit coverage, dispatch contract tests, static
  diagnostics, and full gate.
- [x] [patch] Validate WGPU upload byte size through the shared checked sizing
  helper before buffer initialization, keeping allocation overflow rejection
  consistent across upload, allocation, and download paths. Evidence:
  byte-size overflow unit coverage, static diagnostics, and full gate.
- [x] [patch] Validate binary, unary, and reduction workgroup ranges before
  pipeline setup or intermediate allocation, completing dispatch precheck
  ordering across kernel families. Evidence: workgroup boundary tests,
  contract tests, and full gate.
- [x] [patch] Validate scalar and strided dispatch workgroup ranges before
  transient uniform-buffer acquisition to avoid pool churn on impossible
  dispatch sizes. Evidence: workgroup boundary tests, contract tests, and full
  gate.
- [x] [minor] Make WGPU transient staging/uniform pool acquisition fallible
  with checked alignment arithmetic, routing impossible byte sizes through
  `AllocationFailed`. Evidence: alignment overflow unit tests, contract tests,
  and full gate.
- [x] [patch] Narrow WGPU pipeline-cache mutex scope so shader-module and
  compute-pipeline creation do not run inside the cache critical section.
  Evidence: full dispatch contract suite and benchmark gate.
- [x] [minor] Add typed allocation-failure errors and checked WGPU byte-size
  arithmetic so impossible element counts are rejected before buffer creation
  or copy sizing. Evidence: overflow unit tests, contract tests, and full
  gate.
- [x] [patch] Preallocate reduction intermediate-buffer handle storage from
  the analytically known pass count to avoid vector growth during multi-pass
  command encoding. Evidence: pass-count unit tests, contract tests, and full
  gate.
- [x] [patch] Validate `reduction_with_width` power-of-two block widths before
  empty and singleton fast paths so the documented dispatch contract is
  uniform for every input length. Evidence: boundary contract tests and full
  gate.
- [x] [patch] Add real-adapter `reduction_width` benchmark coverage for
  default vs width-128 reduction dispatch with exact `u32` output validation.
  Evidence: benchmark run and full gate.
- [x] [minor] Thread typed `BlockWidth` through WGPU reduction dispatch via
  `reduction_with_width`, with default `reduction` delegating to
  `BlockWidth::DEFAULT`. Evidence: non-default-width contract test and full
  gate.
- [x] [patch] Name remaining non-test invariant panic sites in default block
  width construction and strided bind slot conversion. Evidence: invariant
  panic scan and full gate.
- [x] [patch] Replace library-code invariant `unwrap()` sites in WGPU
  reduction, pipeline cache, and transient pool locking with explicit
  invariant `expect(...)` messages. Evidence: unwrap scan and full gate.
- [x] [patch] Remove remaining broad negative assertions from the audited Rust
  test scope; absence and mismatch tests now compare concrete values.
  Evidence: assertion-pattern scan and full gate.
- [x] [patch] Replace remaining negative-path existence-only dispatch
  assertions with typed `HephaestusError` contract checks. Evidence:
  WGPU contract and strided tests plus full gate.
- [x] [patch] Reject aliased caller-owned contiguous elementwise output
  buffers before WGPU bind-group creation. Evidence: binary left/right, unary,
  and scalar alias contract tests plus full gate.
- [x] [patch] Make bounded transient WGPU pool reuse best-fit by selecting
  the smallest retained buffer that satisfies a request. Evidence: targeted
  pool regression test and full gate.
- [x] [patch] Store bounded transient WGPU pool entries in `VecDeque` so
  oldest-first count eviction is O(1) instead of shifting retained entries.
  Evidence: targeted pool tests and full gate.
- [x] [patch] Make bounded transient WGPU pools adaptive under count pressure
  by evicting the oldest retained buffer instead of discarding newly recycled
  buffers. Evidence: pool starvation regression test plus targeted pool tests.
- [x] [patch] Bound WGPU transient staging and uniform buffer pools by count
  and retained bytes. Evidence: pure pool unit tests, WGPU contract tests,
  fmt, check, clippy, nextest, doctest, docs, metadata, benchmark, and diff
  checks.
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
- [x] [patch] Consolidate the duplicate GPU→host staging block shared by
  `download` and `download_sub_buffer` into a single private
  `stage_and_read` helper (SSOT for all synchronous device→host readback
  paths). Fixed 4× `as usize` narrowing casts on `u64` byte sizes and 2×
  inline `element_size as u64` patterns, replacing all with `byte_size::<T>`
  checked helper. Evidence: 107/107 wgpu + 21/21 core tests.
- [x] [patch] Consolidate `StagingBufferGuard`/`UniformBufferGuard` (two
  structurally-identical 40-line RAII types in pool.rs) into a single generic
  `PoolBufferGuard<F: Fn(&WgpuDevice, wgpu::Buffer)>` with type aliases.
  Migrated all 12 call-site files to constructor functions `staging_guard` /
  `uniform_guard`. Removed now-unused `crate::UniformBufferGuard` imports.
  Evidence: 107/107 wgpu tests.
- [x] [patch] Extract `encode_elementwise` SSOT in `elementwise/mod.rs` —
  removes 3 structurally-identical 15-line encode-bind-dispatch blocks from
  `binary.rs`, `unary.rs`, and `scalar.rs`. All three `*_into` functions
  now delegate. Evidence: 107/107 wgpu tests.
- [x] [safety] Fix `identity_matrix` and 3× `matpow` allocations using
  unchecked `n * n` / `rows * rows` arithmetic. Both now use
  `checked_mul(...).ok_or(DispatchFailed)` before any allocation.
  Evidence: 107/107 wgpu tests including `linalg_matpow_*`.
- [x] [patch] Normalize workgroup dimension casts in `matmul_into` and
  `batched_matmul_into` to use the shared `to_u32` helper, consistent with
  `kron_into`. Eliminated 3 divergent inline `u32::try_from + format!` sites.
- [x] [patch] Demote `WgpuBuffer::new` to `pub(crate)` and add a
  `debug_assert` validating `len * size_of::<T>() <= buffer.size()`, closing
  the unsound public construction path. Added aliasing semantics doc to Clone.
- [x] [patch] Fix `as usize` casts on test-only `u64` values in
  `pipeline.rs` tests; replaced with `try_into().expect("invariant: ...")`.

## Phase 2: CUDA backend (cuda-oxide + cutile composed) [arch]
- [x] [arch] Gating ADR accepted: `docs/adr/0001-cuda-backend.md` — cuda-oxide
  owns the device substrate (driver/context/streams/memory/transfers, mapping
  one-to-one onto `ComputeDevice`), cutile owns tile/PTX kernel authoring,
  with a strict SoC boundary between them; dynamic driver loading preserves
  no-toolkit-to-compile; adapterless hosts skip like the wgpu suite.
- [x] [arch] `hephaestus-cuda` stage 1: device substrate on cuda-oxide
  (acquisition, typed `PhantomData<T>` buffers, transfers) + contract tests.
- [x] [minor] Stage 2: elementwise/reduction kernels via cutile; stage 3:
  strided variants over the shared packed layout metadata.
- [x] [minor] Differential parity of the CUDA elementwise/reduction dispatch vs
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
  at width 128 vs default. Contiguous elementwise and reduction dispatch now
  also route width through per-width WGSL generation and cache keys.
- [ ] [arch] TPU long-term: `hephaestus-tpu` over the PJRT C API (dynamic
  load, no SDK to compile), only when a consumer drives it; the
  `ComputeDevice` seam already accommodates it. No speculative scaffolding.

## Phase 3: memory + ownership integration [minor]
- [x] [minor] Consume mnemosyne device pools / pinned-host staging (mnemosyne
  Stage D1) for buffer allocation instead of direct device allocation.
- [ ] [minor] melinoe-branded device buffers: ownership transfer across
  host/device/stream as compile-time proofs (melinoe Stage D1 pattern).

## Phase 4: consumers [arch]
- [x] [minor] apollo: `apollo-wgpu-helpers` delegates acquisition to
  `hephaestus-wgpu` with its public API preserved.
- [x] [arch] coeus: re-base GPU backends onto `hephaestus` (coeus MS-60+ Stage D):
  - [x] Re-base `coeus-wgpu` onto `hephaestus-wgpu`.
  - [x] Re-base `coeus-cuda` onto `hephaestus-cuda` once `hephaestus-cuda` is delivered.
- [ ] [minor] moirai: GPU co-scheduling adapter over hephaestus (moirai Stage D).
