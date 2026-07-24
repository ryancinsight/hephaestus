# Backlog — hephaestus

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## HEPH-ROCM-SUBSTRATE-1 [arch] — in-review

- Owner: Codex; scope: new `hephaestus-rocm` crate implementing the existing
  `ComputeDevice`, capability, and acquisition seams with HIP/ROCm device
  acquisition, typed device buffers, transfers, synchronization, topology,
  contract tests, ROCm build/device CI, ADR, README, changelog, and checklist.
  Existing WGPU/CUDA/Metal behavior and Python backend selection are
  non-goals.
- Acceptance: the default workspace compiles without ROCm; the `rocm` feature
  compiles against the pinned HIP bindings on Linux; real HIP allocation,
  zeroing, upload/download, subrange writes, length rejection, capabilities,
  and topology are value-tested on an AMD device; adapterless execution
  returns a typed unavailable error; CI runs a ROCm container build/test lane
  and an explicitly enabled self-hosted AMD hardware lane.
- Claimed files: workspace `Cargo.toml`/`Cargo.lock`; `crates/hephaestus-rocm/**`;
  `.github/workflows/rocm.yml`; `docs/adr/0012-rocm-backend.md`; core/README/
  CHANGELOG/checklist/backlog documentation touched by the new provider.
- Non-goal: HIP kernel authoring and operator-family parity. Re-open as the
  next vertical item when a consumer supplies a ROCm kernel contract.
- Local implementation and package gates pass on 2026-07-24. The new CI
  workflow is YAML-validated. Hosted ROCm run `30097596676` passes the
  container build, feature checks, warning-denied Clippy, Nextest (8/8),
  doctest, and rustdoc at PR head `05300bc`; the manually enabled AMD
  hardware lane remains unexecuted because this host has no ROCm runtime or
  AMD device and no self-hosted runner was available for the pull request.
- Last update: 2026-07-24.

## HEPH-PREPARED-MAP-REDUCTION-1 [minor] — done

- Owner: Codex `/root`; scope: prepared WGPU dot and L2-norm map-reduction
  dispatch, the reduction encoder seam it requires, value/allocation contracts,
  the focused example and benchmark, Rustdoc, changelog, and checklist.
  CUDA behavior, release publication, and unrelated operation families are
  non-goals.
- Acceptance: repeated fixed-buffer dot and L2-norm dispatch reuse pipelines,
  bind groups, metadata, scalar output, and tree scratch; one command encoder
  carries the fused map, reduction, and optional square-root passes; one-shot
  APIs retain their value contract through the same canonical machinery;
  CPU-reference and mutated-input tests pass; allocation identity is pinned;
  the example runs; and a controlled benchmark reports prepared versus
  one-shot dispatch without changing inputs or measurement settings.
- Claimed files: `crates/hephaestus-wgpu/src/application/{linalg,reduction}.rs`,
  their leaf modules if split, `crates/hephaestus-wgpu/tests/contract.rs`, the
  focused example/benchmark and package manifest, `README.md`, `CHANGELOG.md`,
  `CHECKLIST.md`, and this item. Last update: 2026-07-21.
- Current evidence: the two prepared real-adapter value/allocation contracts
  pass (2/2, 1.239 s). The one-shot scalar-reduction tree, prepared dispatch,
  batch dispatch, and fused map-reduction tail now share one prepared plan and
  encoder path. An isolated 65,536-element Criterion comparison measured
  prepared dot 25.7% and prepared L2 23.0% below their one-shot point
  estimates. Local format, all-target Clippy, package/focused Nextest, doctest,
  Rustdoc, example, and benchmark gates pass. PR #60 merged as
  `ff7e77536e7d80b09bba1b88b8c23f85238da608`.

## HEPH-PYTHON-RELEASE-1 [patch] — blocked

- Owner: Codex `/root`; scope: the `hephaestus-python` release workflow,
  protected GitHub environment, distribution documentation, and PyPI trusted
  publisher. Python binding behavior and backend kernels are non-goals.
- Acceptance: a GitHub Release tagged `hephaestus-python-v<version>` builds
  locked Linux, Windows, and universal macOS wheels for CPython 3.9–3.13,
  installs and imports each wheel as `pyhephaestus`, validates Cargo-owned
  distribution identity, attests and attaches the exact artifacts, then
  publishes the same wheels to the `hephaestus-python` PyPI project through
  OIDC.
- Current evidence: the release workflow and synchronized distribution
  contract are implemented, and GitHub environment `pypi` accepts only
  `hephaestus-python-v*` tags. A locked CPython 3.13 wheel builds as
  `hephaestus-python` 0.18.0, installs into an isolated target, and imports as
  `pyhephaestus`. The local GNU linker emits its existing `.drectve` diagnostic;
  the full formatter gate passes after normalizing three pre-existing
  decomposition view expressions. Hosted MSVC and cross-platform CI plus
  pending-publisher registration remain open. Re-open trigger: explicit release
  authority plus PyPI trusted-publisher registration; neither is implied by the
  active provider-development scope.

## HEPH-LAPLACIAN-CONTRACT-1 [arch] — done

- Owner: Codex `/root`; scope: `hephaestus-wgpu` Laplacian parameter contract,
  Leto dependency lock, differential test oracle, ADR and PM artifacts.
- Acceptance: boundary and polarity types have one Leto owner; parameter
  construction delegates dimensional validation to `Laplacian2D`; the local
  CPU stencil is deleted; WGPU results remain differential against the Leto
  CPU implementation; focused package gates pass.
- Evidence: all-target/all-feature check and warning-denied Clippy pass;
  configured Nextest passes 152/152, including eight real-adapter Leto/WGSL
  Laplacian comparisons; doctest and warning-denied rustdoc gates pass.

## HEPH-CUDA-FEATURE-HYGIENE [patch] — done

- Owner: Codex `/root`; scope: CUDA feature-gated infrastructure, pipeline
  keys, and synchronized PM evidence.
- Acceptance: enabling CUDA without decomposition does not compile
  decomposition-only pinned staging storage or pipeline keys, and both feature
  combinations remain warning-clean.
- Evidence: warning-denied all-target Clippy passes for `cuda` and
  `cuda,decomposition`; configured Nextest passes 109/109.

## HEPH-EUNOMIA-0.6-REFRESH [patch] — done

- Owner: Codex `/root`; scope: provider lock and synchronized PM evidence.
- Acceptance: the lock resolves Eunomia 0.6.0 `df77dfde`, Hermes 0.4.0
  `c9bbdf8a`, and Leto 0.39.0 `7afcbd0e`; the all-target/all-feature workspace
  compile and configured provider gates pass.
- Driver: Eunomia E-025c removes the obsolete foreign raw-half numeric/cast
  surface. The initial consumer check proved Hephaestus's Hermes 0.3/Leto 0.38
  lock closure still required that surface.
- Evidence: formatter, all-target/all-feature workspace check, warning-denied
  Clippy, configured Nextest 312/312, doctests, and warning-denied rustdoc pass.

## HEPH-EUNOMIA-0.4-REFRESH [patch] — done

- Owner: Codex `/root`; scope: Eunomia reproducibility pin and synchronized
  provider evidence.
- Acceptance: the lock resolves Eunomia 0.4.0 from its merged default commit;
  the complete warning-denied compile, test, doctest, and rustdoc gates pass.
- Evidence: `Cargo.lock` resolves `49dc115e`; formatter, warning-denied
  all-target/all-feature Clippy, configured Nextest 312/312, doctest, and
  warning-denied rustdoc pass.

## HEPH-EUNOMIA-COMPLEX-1 [arch] — done

- Owner: Codex `/root`; scope: workspace numeric dependency ownership,
  WGPU/CUDA/Metal eigenvalue buffer APIs, Python complex buffer boundary,
  complex provider contracts, and synchronized release/PM artifacts.
- Acceptance: no Hephaestus manifest or source path directly references
  `num-complex`/`num_complex`; typed device buffers and NumPy results use
  `eunomia::Complex`; the Python result path does not allocate a second complex
  vector; all affected package gates pass.
- Driver: Eunomia 0.2.0 PR #36 and ADR 0010.
- Closure evidence: affected package checks and warning-denied Clippy pass;
  supported minimal feature combinations compile; Nextest passes 264/264;
  doctests and warning-denied rustdoc pass; direct residue is zero; and the
  workspace lock pins merged Eunomia commit `34d0cc8a`. Hephaestus PR #48
  merged the provider cutover as `82bb3a7`.

## HEPH-LEGACY-MATH-RESIDUE-1 [patch] — done

- Owner: Codex `/root`; scope: workspace math manifests, WGPU differential
  oracles, and comparative benchmark CPU baselines. The provider owns the
  WGPU/CUDA implementation; this slice deletes only obsolete consumer-side
  reference dependencies.
- Acceptance: `ndarray` and `nalgebra` disappear from Hephaestus manifests,
  tests, and benchmarks; Leto/Leto Ops or analytical value references retain
  differential coverage and real benchmark measurements.
- Last update: 2026-07-17; claim is backed by branch
  `codex/hephaestus-remove-legacy-math` before implementation.
- Closure: direct manifest edges and source references are removed; the
  Leto-only comparative benches and WGPU oracle migration pass the provider
  gates recorded in `gap_audit.md`.

## [HEPH-SCAN-LIMIT-AUDIT] [patch] — done

- Owner: Codex; scope: scan theorem/ADR and synchronized provider PM records.
- Acceptance: determine whether the current one-workgroup tiled scan actually
  hits a line-length workgroup/shared-memory limit before adding a multi-pass
  kernel; record the algebraic bound and a concrete re-open trigger.
- Evidence: both WGPU and CUDA contracts already exercise `L = 513` with
  `BlockWidth::DEFAULT` (`W = 256`), so `L > W` is covered. Each lane loops
  over `ceil(L/W)` values while shared storage remains exactly `W` partials;
  shared-memory use is therefore `O(W)`, independent of `L`. No correctness
  gap justifies a multi-pass implementation in this increment.
- Closure: KS-5b remains a performance follow-up only; reopen when a measured
  device-specific line-length or latency budget is exceeded, with a derived
  floating-point bound for any reordered multi-pass path.

## Closed

- [HEPH-DOWNLEVEL-ACQUISITION-2] [patch] Typed device acquisition preserves
  WGPU's full downlevel descriptor when a consumer raises a mapped
  `DeviceLimits` field. Evidence: exact descriptor-mapping regression,
  warning-denied WGPU Clippy, 137/137 WGPU nextest, doctest, rustdoc, and
  223/223 applicable patch SemVer checks. CFDrs now consumes this contract.

- [HEPH-DOWNLEVEL-LIMITS-1] [minor] `WgpuDevice::downlevel_device_limits`
  exposes the mapped WGPU downlevel limits through `DeviceLimits`. The
  provider's full acquisition-preservation fix is HEPH-DOWNLEVEL-ACQUISITION-2.
  Evidence: value-semantic mapping regression; warning-denied WGPU Clippy;
  136/136 WGPU nextest; rustdoc; doctest; and minor SemVer classification.

- [HEPH-CUDA-BINDGEN-1] [patch] CUDA-enabled builds set `LIBCLANG_PATH` and
  prepend the installed MinGW LLVM directory to `PATH`, replacing the host's
  non-loading UCRT distribution. Evidence: locked `hephaestus-cuda` all-target
  check and the core/WGPU all-target, all-feature check. This closes compilation
  only; CUDA device execution remains independently verified.

- [HEPH-PROVIDER-DEFAULT-2] [minor] Hephaestus 0.15.0 removes every Leto,
  Mnemosyne, Moirai, and Themis revision quarantine, publishes Rust 1.95 from
  every package, and resolves one source identity per provider. Evidence:
  Rust 1.95 focused WGPU check; Rust 1.94.1 resolution rejection; formatting;
  warning-denied release Clippy; release nextest; doctest; rustdoc; and
  196/196 applicable minor semver checks. Driver: Apollo provider convergence.

- [HEPH-STREAM-PREFIX-1] [minor] `CommandStream::copy_prefix` is the provider
  SSOT for bounded device-to-device prefix copies. WGPU and CUDA implement the
  same length-checked contract; the WGPU real-device regression proves the
  destination suffix remains unchanged. Driver: Apollo multilevel Haar DWT.

- [THEMIS-IDENTITY-1] [patch] Themis 0.10 resolves from its default source with
  no workspace-local override.

- [HEPH-EMPTY-001] [patch] CUDA bidiagonal, column-pivoted QR, full-pivot LU,
  Hessenberg, and QR plus WGPU QR now preserve genuine empty dimensions through
  canonical Leto state. CUDA/WGPU value-semantic contracts and the full
  239-test backend suite pass; no synthetic 1x1 factorization remains.

- [WGPU-CB-1] [major] **Superseded by WGPU-ABI-30.** Device construction registers Mnemosyne's
  immutable callback pair before publishing the staging device and surfaces a
  conflicting registration through typed `HephaestusError`. Driver: Mnemosyne
  ADR 0002; local decision: ADR 0005.

## Open

- [x] [minor] HEPH-REQUIRED-FEATURE-1 (owner Codex, completed 2026-07-15;
  scope `hephaestus-wgpu` device acquisition, provider tests, release/PM
  records): `WgpuDevice` now requires a complete `DeviceFeature` set under the
  selected device preference and downlevel-default limits. Driver: Apollo
  native-f16 FFT can require `ShaderF16` without importing WGPU or Pollster.
  Evidence: feature-mapping contract, warning-denied WGPU check and Clippy,
  133-case WGPU nextest run, doctest, rustdoc, and 196/196 applicable
  semver checks against Apollo's 0.13 baseline.

- [x] [patch] HEPH-WGPU-ODD-STORAGE-1 (owner Codex, completed 2026-07-16;
  scope `hephaestus-core` buffer validation, `hephaestus-wgpu` storage and
  transfer implementation, provider tests, ADR/PM records): preserve logical
  odd-length `u16` storage by padding only WGPU's physical byte allocation and
  transfers. Driver: Apollo native-f16 FFT 3x3x3 Bluestein verification.
  Acceptance: exact logical lengths and host values survive upload, write, and
  download; no generic four-byte rejection remains. The focused core/WGPU
  gates, real-device regression, rustdoc, and Apollo consumer integration pass
  in Apollo merge commit `26f433e3`.

- [WGPU-ABI-30] [major] **Review; owner Codex, 2026-07-13.** Migrated the
  provider-owned public WGPU ABI from 26.0.1 to current 30.0.0, update every
  backend call site natively, and prepared Hephaestus 0.13.0 for Apollo. Scope is
  the WGPU dependency and WGPU-consuming crates; the 2026-07-02 `claude-seam`
  claim is stale (clean tree and no scoped commits for more than one day), so
  this item took over only the overlapping WGPU API surface. The complete local
  gate passes; CUDA/Python semver rustdoc is blocked by a cargo-semver-checks
  isolated-build collision in `psm`/`stacker`, while core, Metal, and WGPU
  classification complete. Acceptance and migration design are in ADR 0006.

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
  canonical `lhs`/`rhs` operands. Status: stale claim superseded for
  `device.rs` only by WGPU-CB-1 after no scoped activity since 2026-07-10;
  remaining scope stays with owner claude-seam.
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
  over the seam. **Scan orchestration hoisted** (2026-07-03, commit): the
  duplicated ScanDirection/AxisScanMeta/validation now lives once in
  `hephaestus_core::scan::plan_axis_scan`; backends keep only dialect shader +
  launch (net -212 lines; core gained a std-only leto dep as ADR-0001's shared
  layout vocabulary). **Reduction orchestration parity delivered** (2026-07-05):
  `AxisReductionMeta`, axis-reduction validation, scalar reduction width
  validation, and scalar pass-depth planning now live in
  `hephaestus_core::reduction`; WGPU and CUDA keep only dialect shaders, buffer
  ownership, and launch mechanics. Status: scan done; WGPU/CUDA reduction
  parity done; blocked-decomposition host loops and wrappers remain. The O(L²)
  axis-scan ALGORITHM defect is
  fixed in both backends (2026-07-02): one-thread-per-line sequential scan,
  O(N) total work, combine order preserved so results are bitwise-identical
  to the reference (no test changes); bench 512x4096 f32 axis-1 cumsum
  6.07 ms -> 2.29 ms (2.65x, scan_throughput bench, empirical tier).
- [KS-5b] [minor] Remaining multi-pass tiled scan (block-sums/uniform-add)
  to extend the provider-owned single-workgroup tiled path beyond device
  workgroup/shared-memory limits. Reorders FP addition: needs a derived
  per-element bound encoded in differential tests as a DERIVED tolerance —
  never a widened exact-equality contract. Status: performance follow-up;
  HEPH-SCAN-TILED-1 already handles `L > W` without growing shared storage.
  Re-open trigger: a measured provider workgroup/latency limit or a benchmark
  showing the bounded single-workgroup path misses its declared budget.
- [x] [HEPH-SCAN-TILED-1] [minor] Order-preserving shared-memory tiled scan
  (owner Codex, branch `codex/hephaestus-tiled-scan`, scope
  `hephaestus-core/src/domain/scan.rs`, `hephaestus-wgpu/src/application/scan.rs`,
  `hephaestus-cuda/src/application/scan.rs`, scan contracts and ADR): partition
  each line into contiguous thread chunks, combine chunk totals in logical
  order, and apply the ordered prefix. Acceptance: one workgroup/block per
  line, explicit floating-point reassociation bounds, shared-memory staging
  on both backends, exact integer results, and warning-clean focused gates. The
  multi-pass long-line variant in KS-5b remains a follow-up after this slice.
  Evidence: ADR 0009, core 48/48, WGPU 140/140, CUDA 108/108 with the
  independent concurrent-acquisition abort excluded, and warning-denied
  Clippy for all touched packages.
- [x] [HEPH-CUDA-CONCURRENT-1] [patch] Serialize and memoize CUDA driver
  initialization through a provider-owned `OnceLock` so concurrent
  `CudaDevice::try_default` calls cannot race the dynamic driver loader.
  Acceptance: the existing 16-thread real-device acquisition/transfer
  contract passes without an access violation; missing-driver errors remain
  typed and no test skip or timeout change is introduced. Evidence: the full
  CUDA package nextest now passes 109/109, including the formerly aborting
  16-thread contract, after the provider-owned initialization/context locks.
- [KS-6] [major] `hephaestus-python` module split + domain-logic eviction
  (`split_packed_lu` → core); backend match-arm collapse rides on KS-5.
  Status: in-progress (owner claude-seam; scope `hephaestus-python/**`).
- [KS-7] [minor] Perf batch from the audit: CUDA streams + pinned staging
  (CU-P1/P6/M3), batched-matmul `blockIdx.z` (CU-P5), typed CUDA cache keys
  (CU-P9/P10), wgpu encoder-borrowing batching (WG-P4), fused dot/norms
  (WG-P3), rank/det serial-kernel fix (WG-P1), axis-1 grid-stride reduction
  (WG-P5). Status: todo; criterion baselines before/after each.
  - **CU-P9/P10 done** (commit `8c5d022`, 2026-07-07): replaced the
    per-dispatch `format!()` + `type_name::<Op/T>()` `String` pipeline-cache
    key (15 call sites across `elementwise/{binary,scalar,unary}.rs`,
    `linalg/{kron,matmul,matrix_rank}.rs`, `reduction.rs` ×3, `scan.rs`,
    `sparse/{spmm,spmv}.rs`, `strided.rs` ×3, plus 3 non-generic decomposition
    sites and 3 runtime-authored-kernel sites in `storage_kernel.rs`/
    `stream.rs` found during implementation but not listed in the original
    audit inventory) with a `Copy`, non-allocating `PipelineKey` enum keyed
    on `TypeId` (mirrors `hephaestus-wgpu`'s `(TypeId, TypeId, u32)`
    pattern), one variant per distinct shader family so call sites sharing
    the same `Op: BinaryExpr<CudaC>` concrete types (e.g. binary vs. scalar
    elementwise) can't alias the wrong compiled kernel. This was NOT a
    theoretical risk check — no baseline benchmark was taken (LOW-severity
    mechanism-level fix, no runtime perf claim made); the correctness gate
    was real-hardware verification: 151/151 `hephaestus-cuda` contract tests
    (CUDA) + 295/295 full-workspace tests (CUDA + wgpu) green post-change.
  - **CU-P5 done** (commit `681d3c8`, 2026-07-07): `batched_matmul_into`
    looped `matmul_into` once per batch element — each iteration a separate
    `cuLaunchKernel` plus (Windows, per KS-8) a `cuCtxSynchronize` context
    drain. Added `batched_matmul_kernel` carrying per-operand batch strides
    (broadcast operands pass stride 0) and indexing the batch via
    `blockIdx.z`, so the whole batch dispatches in one launch; batches past
    CUDA's 65535 grid.z hardware cap chunk into further launches via a
    `batch_offset` kernel arg. `batched_matmul_into` had zero prior test
    coverage — added two contract tests with hand-computed oracles
    (non-broadcast 2-batch, and an `lhs`-batch=1 broadcast case) before and
    after the change. New `PipelineKey::BatchedMatmul` variant (own
    shader/entry point) and a `to_i64` stride-conversion helper alongside
    `to_i32`/`to_u32`. Verification: full workspace `cargo nextest run
    --all-features` 297/297 (CUDA + wgpu hardware, up from 295).
  - **WG-P3 already closed** (found 2026-07-07, no code change needed):
    `dot`/`norm_l1`/`norm_l2`/`norm_max` in `hephaestus-wgpu/src/application/
    linalg.rs` already route through the fused `map_reduction`/
    `map_reduction_first_pass` machinery the audit's fix suggested — no
    full-length temporaries are materialized. Stale audit finding, same
    pattern as CU-C1/WG-S1/BOTH-SCAN.
  - **WG-P1 done** (commit `f7537ca`, 2026-07-07): `matrix_properties_with_
    tolerance` dispatched a WGSL kernel at `@workgroup_size(1)`
    `dispatch_workgroups(1,1,1)` — one GPU thread running O(rows·cols²)
    scalar partial-pivoting Gaussian elimination, zero parallelism exploited,
    full pipeline/dispatch/readback overhead paid anyway. Ported the exact
    same algorithm (same pivot order, same `max_abs*tolerance` threshold,
    same sign-flip-on-swap determinant) to run on the host — a
    dispatch-mechanism change, not an algorithm change, verified by the two
    existing contract tests that pin this algorithm's specific divergence
    from Leto's SVD-spectrum criterion
    (`matrix_rank_relative_tolerance_is_the_discriminator`,
    `det_of_near_singular_triangular_is_exact_pivot_product`) still asserting
    the same values. `MatrixRankScalar` now bundles the arithmetic bounds
    (`PartialOrd`, `Neg`/`Sub`/`Mul`/`Div`, `From<f32>`) as supertraits so
    callers still only write `T: MatrixRankScalar`. Deleted the dead WGSL
    shader source, `RankMeta` uniform struct, and `MatrixPropertiesKernel<T>`
    marker. Verification: full workspace `cargo nextest run --all-features`
    297/297 (CUDA + wgpu hardware).
  - **WG-P5 done** (commit `ee92464`, 2026-07-07): the workgroup-parallel
    axis/mean-axis reduction kernels loaded at most one element per lane
    (`if lane < axis_len`), correct only for `axis_len <= width`; dispatch
    fell back to a genuinely serial one-thread-per-row kernel for longer
    axes (zero cross-lane work, full dispatch overhead paid anyway).
    Generalized both kernels to a per-lane strided accumulation loop before
    the existing tree-reduction, correct and fully lane-parallel for any
    axis length — one kernel now covers both regimes; deleted the dead
    serial shader sources and their kernel markers, and the now-unconditional
    dispatch branch. The strided accumulation reassociates the combine
    relative to Leto's sequential CPU reference, so added a new contract
    test at the scale this targets (axis_len=500 > `BlockWidth::DEFAULT`=256,
    real float values) asserting a derived epsilon bound
    (`O(n*eps*sum|x|)` with tree-reduction headroom) rather than exact
    equality; the existing small-integer-fixture test still passes exact
    equality unchanged (integer-valued f32 sums have no rounding error under
    any grouping). Verification: full workspace `cargo nextest run
    --all-features` 298/298 (CUDA + wgpu hardware).
  - **CU-P6/CU-M3 done** (commit `4b8581c`, 2026-07-07): the blocked LU/
    Cholesky decompositions' per-panel host round-trip
    (`download_matrix_region_compact`/`write_matrix_region_compact`) staged
    through a plain `Vec<f32>` (pageable memory, forcing the driver to bounce
    through its own internal pinned staging buffer) with fully synchronous
    per-row `cuMemcpyDtoH_v2`/`cuMemcpyHtoD_v2` calls. Added
    `PinnedHostBuffer<T>` (`cuMemAllocHost_v2`/`cuMemFreeHost` RAII wrapper,
    `Deref`/`DerefMut` to `[T]` so it drops in wherever the `Vec<f32>` was
    used as a slice) — this was CU-M3's "dead capability": zero pinned-memory
    usage existed anywhere in the crate despite host<->device transfers on
    every blocked-decomposition panel round-trip. Switched both functions to
    the pinned buffer and the async copy variants, enqueuing every row before
    one `cuStreamSynchronize` instead of blocking per row. Same-algorithm,
    same-values change (`factor_lu_panel`/`factor_cholesky_panel`'s factored
    output is unaffected). Miri can't execute this crate's CUDA FFI, so the
    new unsafe is verified by real-hardware differential tests instead (the
    existing LU/Cholesky contract tests exercise exactly this code path) —
    stated explicitly rather than claiming Miri coverage it doesn't have.
    Verification: `cargo nextest run -p hephaestus-cuda --features
    cuda,decomposition` 106/106 (real CUDA hardware); full workspace `cargo
    nextest run --all-features` 298/298.
    CU-P1 (async stream pipelining/overlap — the narrower "staging" half of
    the original finding is now closed above) remains open in this item.
    CU-P1's remaining scope (custom per-device `CUstream`s for compute/
    transfer overlap) is lower-value on this crate's primary target
    (Windows/WDDM, where KS-8 already forces a `cuCtxSynchronize` drain
    after every kernel launch) — worth reassessing scope before starting.
  - **WG-P4 closed as a standalone item, re-filed under KS-3** (ADR 0004,
    `docs/adr/0004-wg-p4-composite-op-submit-batching.md`, accepted
    2026-07-08): investigation found the multi-pass reduction tree
    (`reduction_with_width`) already batches its own internal passes into
    one encoder/one submit — `norm_l2`'s "3 submits" is three separately-
    submitting *function calls* chained together (`map_reduction` then
    `unary_elementwise_into` for `sqrt`), not multiplied internal passes.
    Merging them requires giving `reduction_with_width` an encode-into-
    caller's-encoder entry point — real surgery on correctness-load-bearing
    multi-pass logic. This project already has the intended fix for this
    problem class: the `CommandStream`/`GroupedCommandStream` seam
    (KS-2/KS-4/KS-4G), but `CommandStream::encode` requires the newer
    `KernelSource<Dialect>` trait, which `norm_l2`/`map_reduction`/
    `reduction`/the elementwise family don't implement yet — that port is
    KS-3's already-in-progress scope. Decision: defer to KS-3 rather than
    build ad-hoc `encode_*` variants now that KS-3 would make redundant for
    these call sites; re-open WG-P4 independently only if KS-3 stalls or
    excludes this op family.
- [KS-8] [patch] CUDA managed-memory WDDM 0xc0000006 aborts. Status: **done**
  (2026-07-06 focused recheck). The CUDA launch SSOT drains the current context
  with a Windows-gated `cuCtxSynchronize` after each `cuLaunchKernel`, making
  null-stream kernel completion explicit before later host touchpoints. The
  Stage 1 substrate also follows ADR-0001 directly: cuda-oxide initializes the
  driver, creates/binds the context, allocates device memory with
  `cuMemAlloc_v2`, transfers with checked `cuMemcpy*` byte counts, and frees
  with context-bound `cuMemFree_v2`. CUDA allocation hints resolve through one
  non-managed primary-buffer tier: all allocatable placement hints are recorded
  as `MemoryTier::Device`, budget-only tiers are rejected, and
  `MappablePrimaryBuffers` is false. This removes the managed-memory path that
  triggered WDDM `STATUS_IN_PAGE_ERROR` faults. The blocked-decomposition
  region helper uses row-wise 1D copies instead of cuda-oxide 0.4.0's
  Windows-incompatible `CUDA_MEMCPY2D` layout. Evidence: focused live-CUDA
  `cargo nextest run -p hephaestus-cuda
  reduction_sum_matches_cpu_reference reduction_min_max_matches_cpu_reference
  reduction_width_is_part_of_dispatch_contract
  reduction_axis_reduction_generic_matches_cpu linalg_dot_matches_cpu_reference
  linalg_trace_matches_cpu_reference linalg_norms_match_cpu_reference
  hessenberg_reconstructs_and_preserves_similarity_invariants
  non_default_block_width_produces_identical_results` passes 9/9. Residual
  tracking is limited to the documented concurrent-device-acquisition case;
  current focused evidence is `cargo nextest run -p hephaestus-cuda
  concurrent_device_acquisition_is_safe` (1/1).
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

- [x] [minor] Re-export the provider-owned WGPU ABI module from
  `hephaestus-wgpu` as `hephaestus_wgpu::wgpu`. Driver: CFDrs currently
  resolves both direct `wgpu 0.19` and Hephaestus-owned `wgpu 26`; this surface
  lets CFDrs transition raw-kernel boundaries to the provider ABI without
  keeping a separate direct WGPU dependency. Evidence: compile-time re-export
  contract test plus focused WGPU package gate.
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
  parameter block. Follow-up `MultiStorageDevice` provides the backend-owned
  `storage_binding(binding, &D::Buffer<T>)` constructor, so downstream structs
  can stay generic over the device while each backend keeps its native binding
  representation. Driver: Kwavers 3-D static DAS (five bindings) and
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
