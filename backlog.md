# Backlog — hephaestus

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## Open

ADR-0004 kernel-seam programme (atlas `docs/adr/0004-hephaestus-kernel-seam.md`,
audit `docs/audit/2026-07-02-hephaestus-gpu-substrate-audit.md`; branch
`arch/kernel-seam`, owner claude-seam session 2026-07-02):

- [KS-1] [minor] Core dialect + op vocabulary (`KernelDialect`, `DialectScalar`,
  `UnaryExpr`/`BinaryExpr`/`CombineExpr`, `OpIdentity`/`IdentityToken`, ZST
  markers). Status: **done** (commit `2c01d36`).
- [KS-2] [minor] Authored-kernel seam in core (`KernelInterface`,
  `KernelSource<L>`, `KernelDevice` with `BindingHandle`/`Prepared`/`Stream`
  GATs, `CommandStream`, `Binding`, `validate_bindings`). Status: **done**
  (commit `f18bb72`).
- [KS-3] [major] Backends consume the core op vocabulary; per-backend trait
  pairs and duplicated ZSTs deleted; CUDA binary/scalar templates renamed to
  canonical `lhs`/`rhs` operands. Status: in-progress (owner claude-seam;
  scope `hephaestus-wgpu/**` + `hephaestus-cuda/**`).
- [KS-4] [minor] `KernelDevice`/`CommandStream` impls for `WgpuDevice` and
  `CudaDevice` + shared generic contract tests. Supersedes the standing "CUDA
  implementor for multi-storage kernels" item below for NEW consumers; the
  existing storage-kernel trio stays until kwavers/apollo migrate to the
  authored seam (removal then is [major]). Status: **done for WGPU and CUDA**.
  WGPU evidence: `cargo fmt -p hephaestus-wgpu --check`, `cargo check -p
  hephaestus-wgpu`, `cargo clippy -p hephaestus-wgpu --all-targets --no-deps --
  -D warnings`, and `cargo nextest run -p hephaestus-wgpu stream` pass 5/5.
  CUDA evidence: `cargo fmt -p hephaestus-cuda --check`, `cargo check -p
  hephaestus-cuda`, `cargo clippy -p hephaestus-cuda --all-targets --no-deps --
  -D warnings`, `cargo clippy -p hephaestus-cuda --no-default-features
  --all-targets --no-deps -- -D warnings`, `cargo nextest run -p
  hephaestus-cuda stream` pass 3/3, and `cargo nextest run -p hephaestus-cuda
  --no-default-features stream` pass 3/3.
- [KS-4G] [minor] Grouped authored-kernel seam for consumers with multiple WGPU
  bind groups, flat CUDA argument lists, and same-region ordered sequences.
  Status: **done for WGPU and CUDA**.
  Evidence: `cargo fmt -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda
  --check`, `cargo check -p hephaestus-core`, `cargo check -p hephaestus-wgpu`,
  `cargo check -p hephaestus-cuda --no-default-features`, `cargo check -p
  hephaestus-cuda`, `cargo clippy -p hephaestus-core -p hephaestus-cuda
  --all-targets --no-default-features --no-deps -- -D warnings`, `cargo clippy
  -p hephaestus-wgpu --all-targets --no-deps -- -D warnings`, `cargo nextest
  run -p hephaestus-wgpu stream` pass 8/8, and `cargo nextest run -p
  hephaestus-cuda --no-default-features stream` pass 6/6. Driver: Kwavers PSTD
  no longer needs a missing provider seam for multi-group WGPU/CUDA authored
  kernels or same-pass WGPU timestep sequencing; remaining PSTD work is
  consumer shader/ABI migration and CUDA C source authoring.
- [KS-4D] [minor] Device acquisition policy vocabulary. Status: **done**.
  `hephaestus-core::DevicePreference` now carries backend-neutral
  high-performance vs low-power selection, and `hephaestus-wgpu` maps it to
  WGPU only inside provider constructors. Driver: Kwavers removed
  `wgpu::PowerPreference` from public GPU device creation and PSTD/beamforming
  acquisition call sites.
- [KS-4C] [minor] Device capability vocabulary. Status: **done**.
  `hephaestus-core::DeviceFeature` and `DeviceLimits` now carry backend-neutral
  optional capability and compute-limit reporting. `ComputeDeviceCapabilities`
  is the trait-level seam for querying those values generically.
  `hephaestus-wgpu` maps the vocabulary at the WGPU provider boundary, and
  `hephaestus-cuda` now maps real CUDA driver attributes into the same contract
  without fabricating WGPU-only storage-binding limits. Driver: Kwavers removed
  `wgpu::Features` and `wgpu::Limits` from public `GpuDevice` capability APIs
  and made its backend contexts generic over `D: ComputeDeviceCapabilities`.
- [KS-5] [major] Per-family host-orchestration consolidation into core generic
  over the seam — scan first, then blocked decompositions, then wrappers.
  Status: todo (orchestration hoist). The O(L²) axis-scan ALGORITHM defect is
  fixed in both backends (2026-07-02): one-thread-per-line sequential scan,
  O(N) total work, combine order preserved so results are bitwise-identical
  to the reference (no test changes); bench 512x4096 f32 axis-1 cumsum
  6.07 ms -> 2.29 ms (2.65x, scan_throughput bench, empirical tier).
- [KS-5b] [minor] Tiled shared-memory scan (Hillis-Steele/Blelloch single-tile
  fast path + block-sums/uniform-add multi-pass) to restore full-width
  parallelism on long lines. Reorders FP addition: needs the derived
  per-element bound ~O(log2(L)*eps*sum|x|) (tree depth log2 L vs sequential
  depth L) encoded in the differential tests as a DERIVED tolerance for the
  reordered kernel — never a widened exact-equality contract. Status: todo.
- [KS-6] [major] `hephaestus-python` module split + domain-logic eviction
  (`split_packed_lu` → core); backend match-arm collapse rides on KS-5.
  Status: in-progress (owner claude-seam; scope `hephaestus-python/**`).
- [KS-7] [minor] Perf batch from the audit: CUDA streams + pinned staging
  (CU-P1/P6/M3), batched-matmul `blockIdx.z` (CU-P5), typed CUDA cache keys
  (CU-P9/P10), wgpu encoder-borrowing batching (WG-P4), fused dot/norms
  (WG-P3), rank/det serial-kernel fix (WG-P1), axis-1 grid-stride reduction
  (WG-P5). Status: todo; criterion baselines before/after each.
- [KS-8] [patch] CUDA managed-memory root cause: 9 tests abort deterministically
  with OS 0xc0000006 (STATUS_IN_PAGE_ERROR) on this machine's real hardware
  (concurrency, trace, dot, norms, hessenberg, 3× reduction, strided
  block-width) — A/B-verified pre-existing on unmodified baseline; all share
  the reduction-kernel/managed-memory pathway. Suspect: in-band host-written
  allocator metadata on `cuMemAllocManaged` pages + `cuMemAdvise` hardcoding
  ordinal 0 (audit CU-P7). Fix direction: real `cuMemAlloc` device tier in
  mnemosyne + advise the actual ordinal. Status: todo; blocks the CUDA
  full-green claim.
- [KS-9] [minor] `hephaestus-metal` decision: 1,276-line pure-forwarding crate
  over wgpu-Metal — reduce to `WgpuDevice::try_metal` constructor ([major]
  break) or record the alias-crate justification. Status: todo (user decision
  useful on the break).

- [arch] Add a concrete CUDA implementor for multi-storage beamforming kernels
  when a CUDA beamforming kernel exists. The backend-neutral trait and WGPU
  implementation are delivered; remaining work is the CUDA kernel/launch
  implementation and downstream Kwavers verification against that provider.
  (For new consumers this is subsumed by KS-4's authored-kernel seam.)

## Delivered

- [x] [patch] Add provider-owned WGPU capability accessors. `WgpuDevice` now
  exposes `features()` and `limits()` so consumers can report capabilities
  without borrowing raw `wgpu::Device` handles. Driver: Kwavers backend contexts
  removed public raw device/queue accessors and use these accessors for
  capability reporting. Evidence: Hephaestus fmt/check/clippy plus downstream
  Kwavers check/clippy/nextest backend-device-multi_gpu filter passing 34/34.
- [x] [minor] Add backend-neutral partial device-buffer writes to
  `ComputeDevice`. WGPU, CUDA, Metal, and the CUDA-unavailable stub now satisfy
  `write_sub_buffer` through the provider trait, with contract tests covering
  partial overwrite preservation, out-of-range rejection, and empty tail writes.
  Evidence: focused fmt/check/clippy and `cargo nextest run -p hephaestus-wgpu
  -p hephaestus-cuda -p hephaestus-metal --no-default-features
  write_sub_buffer` passing 9/9.
- [x] [patch] Complete the remaining `hephaestus-wgpu` consumer migration from
  deleted backend-local shader traits to shared `hephaestus_core` dialect
  traits. Linalg, random, sparse, scan exports, and crate exports now use
  `DialectScalar`, expression traits, and typed identity traits; no
  compatibility aliases were reintroduced. Evidence: stale-name source audit
  clean, `cargo check -p hephaestus-wgpu`, and `cargo clippy -p
  hephaestus-wgpu --all-targets --no-deps -- -D warnings`.
- [x] [patch] Remove the stale `DeviceExt` import from `hephaestus-wgpu`
  storage-kernel dispatch so downstream provider builds stay warning-clean.
  Evidence: `cargo check -p hephaestus-wgpu`.
- [x] [minor] Implement `KernelDevice`/`CommandStream` for `WgpuDevice`.
  Authored WGSL kernels now prepare through the shared `KernelInterface` /
  `KernelSource<Wgsl>` contract, encode ordered dispatch/copy/zero-fill streams,
  validate typed binding layouts, and submit via the provider boundary. Evidence:
  `cargo fmt -p hephaestus-wgpu --check`, `cargo check -p hephaestus-wgpu`,
  `cargo clippy -p hephaestus-wgpu --all-targets --no-deps -- -D warnings`, and
  `cargo nextest run -p hephaestus-wgpu stream` pass 5/5.
- [x] [minor] Implement `KernelDevice`/`CommandStream` for `CudaDevice`.
  Authored CUDA C kernels now prepare through the shared `KernelInterface` /
  `KernelSource<CudaC>` contract, encode ordered dispatch/copy/zero-fill
  streams, validate typed binding layouts, and submit through the CUDA provider
  boundary. Evidence: `cargo fmt -p hephaestus-cuda --check`, `cargo check -p
  hephaestus-cuda`, `cargo clippy -p hephaestus-cuda --all-targets --no-deps --
  -D warnings`, `cargo clippy -p hephaestus-cuda --no-default-features
  --all-targets --no-deps -- -D warnings`, `cargo nextest run -p
  hephaestus-cuda stream` pass 3/3, and `cargo nextest run -p hephaestus-cuda
  --no-default-features stream` pass 3/3.
- [x] [minor] Add grouped authored-kernel dispatch for WGPU/CUDA consumers that
  require multiple WGPU bind groups and same-region sequencing. Core now exposes
  `GroupedKernelInterface`, `GroupedKernelSource`, `GroupedKernelDevice`,
  `GroupedCommandStream`, `GroupedKernelSequence`, `GroupedBindingDecl`, and
  `GroupedBinding`; WGPU builds one bind group per declared group and can encode
  an ordered grouped sequence inside one compute pass, while CUDA launches the
  same contract as a flat ordered argument list on the bound stream. Driver:
  Kwavers PSTD field/kspace/sensor/absorption kernels can migrate to a
  Hephaestus provider trait rather than a local raw-WGPU helper. Evidence:
  focused fmt/check/clippy plus WGPU and CUDA stream nextest filters.
- [x] [minor] Add backend-neutral device synchronization to `ComputeDevice`.
  WGPU, CUDA, and Metal now expose explicit completion through the provider
  trait (`Device::poll`, `cuCtxSynchronize`, and Metal's WGPU delegation),
  allowing downstream crates to request blocking transfer semantics without
  importing a concrete GPU API. Driver: Kwavers visualization `DataPipeline<D>`
  uses this with generic provider buffers instead of raw WGPU queue/poll
  ownership. Evidence: Hephaestus check/fmt/clippy/nextest plus downstream
  Kwavers visualization check/clippy/nextest and data-pipeline source audit.
- [x] [minor] Add backend-neutral multi-storage kernel dispatch for downstream
  kernels wider than unary/binary storage layouts. `MultiStorageKernel<D, P, B>`
  carries the generic provider contract; `WgslMultiStorageKernel` and
  `WgslStorageBinding` own the real WGPU shader, bind-group layout, uniform
  buffer, encoder, and submission path for N storage buffers plus one POD
  parameter block. Driver: Kwavers 3-D static DAS (five bindings) and
  dynamic-focus DAS (seven bindings) now bind through this provider path without
  a Kwavers local helper. Evidence: Hephaestus check/clippy/nextest and
  downstream Kwavers 3-D beamforming check/clippy/nextest.
- [x] [minor] Add backend-neutral unary and binary storage-kernel dispatch
  contracts for downstream WGPU/CUDA-generic consumers. `DispatchGrid`
  centralizes checked workgroup coverage arithmetic,
  `UnaryStorageKernel<D, T, P>` and `BinaryStorageKernel<D, T, P>` bind kernels
  to `ComputeDevice` buffers without exposing a concrete GPU API, and
  `WgslUnaryStorageKernel` / `WgslBinaryStorageKernel` supply the real WGPU
  dispatch implementations. Evidence: focused core kernel nextest (2/2), fmt,
  and core/wgpu compile checks.
- [x] [minor] Add dynamic-rank `hephaestus-cuda` strided elementwise entry
  points over borrowed shape/stride slices so runtime-shaped consumers such as
  Coeus can delegate rank <= 4 strided CUDA primitive binary/unary kernels to
  Hephaestus. Static-rank and dynamic-rank APIs now share the same private launch
  helpers. Evidence: focused `hephaestus-cuda` strided nextest (11/11), clippy,
  rustdoc, and downstream Coeus CUDA live parity (69/69).
- [x] [patch] Give strided scalar ops a dedicated pooled-uniform kernel
  (`StridedScalarKernel`) so a strided scalar dispatch no longer allocates +
  uploads a one-element device storage buffer per call (matches the contiguous
  `scalar_elementwise_into` SSOT). Benefits `hephaestus-metal` via delegation.
  Evidence: `strided_scalar_matches_binary_broadcast_semantics` (value-identity),
  full workspace gate, clippy `-D warnings`.
- [x] [patch] Eliminate per-panel host-buffer allocations in blocked
  Cholesky/LU/QR. Added the region-download SSOT
  `download_matrix_region_compact_into(out: &mut Vec)` (reuses host capacity),
  removed the dead returning-`Vec` `_reusable` wrapper, and hoisted each
  decomposition's per-panel host scratch above the loop (LU: `col_panel`,
  `row_panel`, `diag`; QR: `panel`, `packed_vectors`, `vector_offsets`; Cholesky:
  `panel`). Removes `O(n/b)` host allocations per call. Evidence: blocked
  Cholesky/LU/QR cross-block-boundary contract tests + full 230-test workspace
  gate; clippy `-D warnings`.
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
- [x] [minor] Placement-aware allocation: thread themis `PlacementHint` /
  `MemoryTier` (Hbm, Gddr, HostPinned, unified) through `ComputeDevice`
  allocation so consumers select device-memory tiers explicitly. wgpu maps the
  hint to buffer usages (HostPinned → mnemosyne-staged host-mapped MAP buffer;
  device tiers → STORAGE); CUDA maps to the device / host-pinned / unified
  mnemosyne backends. Value-semantic coverage closed the prior tier-field-only
  gap: `test_placement_aware_allocation` now verifies Dram and Device uploads
  and zeroed allocations round-trip data, while HostPinned asserts tier/length
  (the persistently host-mapped staging buffer is read via its mapped pointer,
  not `download` — a queue submit touching a mapped buffer is a wgpu error).
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
