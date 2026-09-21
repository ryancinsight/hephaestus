# Checklist — hephaestus

## review_gpu — CUDA scalar products

- [x] Collect required-device value/PTX and strict Clippy results for [scalar products](backlog.md#heph-cuda-dense-product-scalars).
- [x] Run the full configured host/CUDA gates and public migration SemVer check for [scalar products](backlog.md#heph-cuda-dense-product-scalars).
- [ ] Integrate independent review, commit and publish [scalar products](backlog.md#heph-cuda-dense-product-scalars) through the repository merge gate.

Sprint target: 0.18.0. Phase: Closure.

## HEPH-FFT-PROVIDER-1 [minor] [arch] [perf] — Owner: Codex

Readback-completion PR #232 merged as `b4e170e` from exact reviewed head
`cf0907e`. Provider PR #230 merged as `48bb731` and device-preflight PR #231
merged as `1636301`. Apollo's warm STFT phase
census isolated `stage_and_read`'s per-call completion channel. The candidate
uses eight fixed slots acquired before submission; reader/callback ownership
quarantines pending state through error, cancellation, and unwind, while
capacity overflow allocates before submission. Deterministic capacity-plus-one,
terminal-state, cancellation, and reuse tests pass. The bounded one-slot Loom
model covers concurrent final-owner release, callback completion publication,
cancellation, overflow, and racing reuse (1/1 in 0.671 seconds). Warning-denied
all-target Clippy, exact-candidate WGPU nextest (28/28 in 25.096 seconds),
doctests (2/2), and rustdoc pass. Independent exact-head review is GREEN;
hosted lockfile and host verification pass. CUDA, ROCm, Metal, and WGPU hosted
checks were still running when the PR merged and remain a collection watchpoint.
Apollo and Kwavers consumer closure remains.

Retained-parameter increment: `WgpuBoundGroupedDispatch::update_params`
mutably rewrites the uniform through the owning queue while preserving fixed
buffers, bind groups, and launch geometry. Exact output changes with logical
length and arithmetic parameters through the same retained dispatch;
foreign-device mutation fails before writing. Warning-denied all-target
Clippy, WGPU Nextest (28/28 in 31.381 seconds), Rustdoc, rustfmt, and diff
checks pass. `cargo-semver-checks` did not reach API comparison because its
local baseline clone failed on an oversized packed-object entry. Independent
exact-candidate review of `e6218da` is GREEN. Lease: none; exact-lock hosted
gates and merge remain.

- [x] Audit Leto, Apollo, Hephaestus, and Kwavers ownership, dimensionality,
      duplicated WGPU kernels, dependency direction, and existing conformance.
- [x] Record the provider ownership, rank-generic contract, staged cutover, and
      no-fallback rule in ADR 0053.
- [x] Add the device-neutral split-complex FFT plan, validation, and prepared
      operation seam to `hephaestus-core` with boundary/adversarial tests.
- [x] Port and consolidate WGPU radix/Bluestein execution into
      `hephaestus-wgpu`, extending the current 3-D-only consumer surface to the
      same 1-D/2-D/3-D implementation.
- [ ] Add generic analytical, round-trip, Apollo/Leto differential, real-device,
      allocation, and matched performance coverage.
      Direct analytical, round-trip, real-device, prepared-resource, and
      bounded benchmark-smoke coverage is complete. The benchmark calls the
      provider-neutral encode seam and checks independent forward DFT bins with
      a depth-derived error bound. Prepared plans own fixed operand handles;
      static inspection and an in-pass real-device regression prove the removal
      of duplicate volumes and warm copies. Apollo/Leto differential and
      consumer-boundary allocation instrumentation remain; provider scalar
      timing is complete. The generic f32/f16 suite passes 22/22 focused FFT
      tests in 17.081 seconds. Grouped dispatch preserves command, workspace,
      and root-table identities, while source inspection excludes
      Hephaestus-owned warm
      allocation, compilation, bind construction, transfer, and copy but not
      opaque driver allocations.
      Provider PR #222 merged as `cfadc373`; the next provider increment adds a
      matched 256x128x128 repeated-pair workload against Kwavers's measured
      10.09 ms/step pre-cutover baseline. The staged plan measured 13.810 ms and
      was rejected. The fused workgroup radix measures 7.9974 ms (42.1% lower),
      emits one dispatch per active axis, removes 64 MiB of paired-plan workspace,
      and passes direct-oracle rank plus singleton/no-workspace tests. The full
      required-device WGPU package passes 233/233 with no skips in 62.109
      seconds; the separately tracked test-topology item owns reducing that
      runtime without weakening coverage or budgets. The full Kwavers step
      remains the deletion gate.
      The native-scalar increment's complete package run passes 242/242 with no
      skips in 76.936 seconds. This exceeds the ordinary package budget and
      refreshes `HEPH-WGPU-TEST-DEVICE-REUSE-1`'s entry baseline; no timeout or
      workload was changed. The selected-axis increment adds direct-DFT and
      inverse row oracles for `[3, 8]` and `[3, 5]`, rejection before mutation
      for empty, duplicate, and out-of-range selections, and retained grouped
      binding reuse. Focused real-device Nextest passes 14/14 in 15.486 seconds;
      warning-denied all-target Clippy, formatting, doctests, rustdoc, core/WGPU
      SemVer (196/196 each), and two independent static reviews are green.
- [x] Generalize the existing rank-generic WGPU FFT implementation to native
      `f16` through `FftOps<D, T>` without a second plan family; reject devices
      without shader-f16 support and add shared f32/f16 analytical,
      round-trip, allocation, and dispatch-residency coverage.
      One sealed scalar contract now instantiates the existing plan, strategy,
      and WGSL family for f32 and native binary16. Missing `ShaderF16` fails
      before allocation or mutation with no fallback. Roots and reciprocal
      scales are precomputed in the selected scalar; the compact staged table
      reconstructs radix-four's second half-circle without another allocation.
      RTX 5080/Vulkan paired forward/inverse Criterion regression-slope time
      estimates (95% confidence intervals) are 162.71 microseconds
      [161.25, 164.67] versus 171.31 [168.20, 174.41] at 65,536 elements and
      221.64 [219.22, 225.42] versus 215.52 [212.69, 218.68] at 64 cubed. This
      does not support a universal binary16 speed claim. The benchmark validates
      sampled forward bins against an independent direct DFT and rejects
      identity output. Warning-denied all-target Clippy, Windows AArch64,
      doctests, rustdoc, core/WGPU SemVer (196/196 each), benchmark smoke, and
      independent review are green.
- [ ] Migrate Apollo and Kwavers, add closed `Leto`/`Hephaestus` selection at
      the Kwavers operation boundary, and delete both consumer-owned WGPU FFT
      implementations.
- [ ] Pass independent architecture review, focused/full gates, exact-head
      hosted provider and consumer CI, then merge dependency-order commits.
      The local provider architecture/performance and selected-axis/binding
      re-reviews are green; hosted exact-head and consumer closure remain.

## HEPH-BOOK-REGROUND-1 [patch] [docs] — Owner: current Atlas session

- [x] Compare the affected chapters with the current core device, buffer,
      capability, operation, decomposition, and backend APIs.
- [x] Remove fabricated identifiers, stale dynamic-buffer claims, and
      unsupported cross-provider ownership statements from the affected book
      chapters and README.
- [x] Pass the formatter, locked hephaestus-host build, warning-denied Clippy,
      host Nextest (2/2), mdBook build, and strict internal-link checking.
- [ ] Collect a clean hosted mdBook test result and reconcile the parallel
      host-backend chapter closeout before merging.

## HEPH-BOOK-TEST-2026-08-20 [patch]

- [x] Diagnose PR #214 job 96544627958: package build passed; mdBook
      compilation failed because included examples lacked explicit extern crate
      declarations for their staged crates.
- [x] Add declarations for Hephaestus Core/Host and Themis and repin the
      shared workflow to Atlas 20c9398.
- [x] Pass local formatting, mdBook build, strict links, and diff checks.
- [ ] Collect the exact-head hosted rerun, then merge and verify the default.

## HEPH-FDTD-PROVIDER-1 — Owner: Codex

- [x] Add validated provider-neutral 3D FDTD geometry, medium, and velocity
      storage types.
- [x] Add the `Fdtd3dOps` provider seam with explicit in-place step ordering.
- [x] Implement WGPU central-difference velocity and pressure kernels with
      spacing and medium coefficients owned by the provider.
- [x] Add invalid-storage and independent one-step value contracts.
- [x] Synchronize ADR, gap audit, changelog, and board records.
- [ ] Pass exact-head hosted device-required WGPU CI.
- [ ] Migrate Kwavers and close the cross-repository integration item.

## HEPH-METAL-ACQUISITION-1 [minor] — Owner: Codex

- [x] Record the current Metal acquisition baseline and WGPU substrate closure.
- [x] Add Metal-only feature/limit-aware single and bounded multi-device
  acquisition without fallback to another WGPU backend.
- [x] Implement `ComputeDeviceAcquisition` for `MetalDevice` and add
  value-semantic capability/boundary contracts.
- [x] Run focused format, Nextest, warning-denied Clippy, doctest/Rustdoc,
  independent review, SemVer classification, and exact-head provider CI.

Implementation owner: Codex on `codex/feat-metal-device-acquisition`. Claimed
files are WGPU device acquisition, Metal device infrastructure and contract
tests, plus owner-keyed `CHANGELOG.md`, `backlog.md`, `checklist.md`, and
`gap_audit.md` entries. KS-5 decomposition files remain excluded.
Baseline exact-source Nextest run `b8f7b647-4d6e-48fe-8cd1-03a8d5e94eaa`
passes 1/1. Final focused WGPU run `a875a543-9f52-440b-a648-b490be28c166`
passes the preference policy 1/1; Metal run
`510d5e60-eefc-4dd8-a395-09bd6bfc3b9d` passes the zero-bound, shared
acquisition, and capability contracts 3/3. WGPU and Metal all-target
compilation, warning-denied Clippy, doctests, warning-clean Rustdoc, and
formatting pass. WGPU SemVer passes 196/196 applicable checks; Metal SemVer is
blocked before API analysis by an unexpected MSVC linker failure in the tool's
temporary rustdoc graph, which was deleted with its 9.0 GiB repo-local cache.
Physical adapter selection, optional-feature intersection, and required-limit
enforcement remain native macOS CI evidence from this Windows host. Independent
re-review approves with no remaining findings.
Exact implementation head `3b8cc85` passes WGPU run `30762519498`, CUDA run
`30762519504`, ROCm run `30762519499`, and native macOS Metal run `30762519503`.
Hardware-only NVIDIA and AMD jobs skip because this dispatch did not request
self-hosted devices.

## HEPH-MOIRAI-PACKAGE-1 [patch] — Owner: Codex `/root`

- [x] Bind the `moirai` Rust import to package `moirai-runtime`.
- [x] Refresh the lockfile and pass clean locked metadata plus the focused WGPU
  package check.
- [x] Align the checkout-local CI patches with Mnemosyne's published packages.
- [x] Version every normal path and Git dependency used by publishable packages.
- [ ] Pass all six exact-source crates.io dry runs.
- [ ] Merge the provider identity fix before refreshing Apollo and Coeus.

## HEPH-STATEFUL-ZERO-LR-1 [patch] — Owner: Codex

- [x] Admit finite zero learning rates without relaxing epsilon or domain
      validation.
- [x] Cover all five parameter contracts with focused Nextest.
- [ ] Pass warning-denied Clippy and exact-head hosted checks; merge.

## HEPH-ATTENTION-PROVIDER-1 [minor] [arch]

- [x] Define the device-neutral rank-3 attention operands, mask, gradients,
      planning contract, and prepared dispatch seam.
- [x] Implement WGPU, CUDA, ROCm, and Metal provider-owned attention without
      host execution or provider fallback.
- [x] Instantiate one Leto-differential conformance suite across available
      backends and supported scalars.
- [ ] Cut Coeus CPU dispatch directly to Leto and accelerator dispatch directly
      to Hephaestus; delete superseded local kernels and fallbacks.
- [x] Pass provider-side warning-denied, focused contract, and exact-head
      CUDA/ROCm/WGPU/Metal gates; source `702eba8` merges at default `4714b8c`
      with runs `32026666522`, `32026666500`, `32026666544`, and
      `32026666549`.

Implementation owner: Codex on `codex/hephaestus-attention-provider`.
Claimed files are ADR 0040; the Hephaestus core attention domain; WGPU, CUDA,
ROCm, and Metal attention application modules; shared attention conformance;
the owner-keyed PM entries; and the subsequent disjoint Coeus attention cutover.
Local provider gates pass: core attention 6/6, combined WGPU/CUDA attention
11/11, and ROCm no-default-feature attention 9/9, with warning-denied checks.
The broader touched-package Nextest lane passes 465/465 and all six touched
crates pass doctests. `hephaestus-core` passes 196/196 SemVer checks against
`origin/master`. The remaining closure is the direct Coeus cutover and
exact-head hosted gates.
Independent review found that empty ROCm forward dispatch returned before its
status reset; the reset now precedes the empty guard and the focused ROCm lane
passes 9/9 after the correction.

## HEPH-WGPU-QR-TAIL-SYNC-1 [minor] [perf]

- [x] Add one backend-neutral allocation-free packed-panel Householder apply.
- [x] Gather and write the final panel/tail pair through one encoder,
      submission, and readback poll.
- [x] Preserve complete `R` and solve contracts across 32/33/35/64/65 columns.
- [x] Record a matched Criterion before/after result and the bounded staging
      memory tradeoff.
- [x] Run warning-denied, doctest, semver, benchmark-smoke, and exact-head
      WGPU, CUDA, ROCm, and Metal CI gates.

Implementation owner: Codex on `codex/hephaestus-wgpu-qr-tail-sync`; ADR 0038.

Local evidence: all-target core/WGPU compilation passes. Focused Nextest passes
8/8 in 4.976 seconds, including the packed-panel analytical differential and
all blocked-QR contracts. Criterion 0.8.2 measures the unchanged 70×35 workload
at 516.58 µs before and 346.19 µs after, a 32.984% median reduction. Criterion's
central change estimate is −29.612% (95% interval −34.402% to −23.972%,
`p = 0.00`).
Persistent compact device and host algorithm scratch capacities are unchanged;
the paired readback holds one extra 840-byte staging buffer until the shared
poll completes. Warning-denied core/WGPU all-target Clippy and formatting pass.
The final full package Nextest and doctest attempts are blocked before
Hephaestus execution by fresh peer-owned Aequitas work: `PerCubicMeter`
implements both `NumberDensity` and its equivalent `ReciprocalVolume` alias
(`E0119`). The earlier exact focused run predates that unrelated tree shift;
hosted locked provider CI supplies the full execution gate. Cargo-semver-checks
0.48.0 compares `hephaestus-core` with `origin/master`: 196 checks pass and 57
inapplicable checks skip, with no semver update required. PR #142 exact-head
provider jobs pass: CUDA 90444185125 (6m21s), ROCm 90444185117 (5m52s), WGPU
90444185164 (6m39s), and macOS Metal 90444185226 (7m53s). Hardware-only CUDA
and ROCm jobs correctly skip on the pull-request event.

## HEPH-PYTHON-RELEASE-1 [patch]

- [x] Add the pinned build-once GitHub Release and PyPI workflow.
- [x] Document the `hephaestus-python` distribution, `pyhephaestus` import,
      Cargo version source, supported CPython range, and OIDC publication
      contract.
- [x] Build, install, import, and inspect a production CPython 3.13 wheel
      locally as `hephaestus-python` 0.18.0 / `pyhephaestus`; the local GNU
      linker retains its existing `.drectve` diagnostic pending hosted MSVC
      coverage.
- [x] Create the protected `pypi` environment restricted to
      `hephaestus-python-v*` tags.
- [ ] Pass hosted CI on the exact release-automation head.
- [ ] Register the PyPI pending trusted publisher.

## HEPH-SCAN-LIMIT-AUDIT [patch]

- [x] Audit the WGPU/CUDA scan shader and planner bounds before adding a
      multi-pass implementation.
- [x] Confirm the existing `L = 513`, `W = 256` integer contracts prove the
      `L > W` path, while shared partial storage remains `W` elements.
- [x] Record the theorem, evidence tier, and measured re-open trigger in the
      provider backlog, gap audit, and ADR 0009.

Evidence tier: source algebra plus existing value-semantic real-device
contracts. No multi-pass code is added because the stated workgroup/shared-
memory limit is not present in the current implementation.

## Superseded — WGPU-CB-1 immutable staging callbacks [major]

Superseded: the immutable staging-callback design this 696-line checklist
tracked was replaced by the provider-owned transfer surface (the owned-readback
items in `gap_audit.md`). Its increment log — including the 2026-07-06 KS-8 WDDM
launch-drain recheck it absorbed — is a per-PR record; recover any of it with
`git log --grep='Item: WGPU-CB-1'` or `git log -p -- checklist.md`.

- [ ] Semver gate: make the baseline clone resolve repository-external sibling
  path dependencies. The current 0.12.0 rustdoc build passes, but the baseline
  clone cannot find `../leto/crates/leto`; the local Atlas graph is green with
  Moirai's committed Mnemosyne 0.2 requirement and no consumer-tree edits.

## Unreleased WGPU Leto parity linalg [minor]

The WGPU Leto-parity linear-algebra surface is delivered and differential-tested
against Leto, `ndarray` and a `nalgebra`-backed reference: allocating and
caller-owned `matmul`/`batched_matmul`, `dot`, `trace`, `norm_l1`, `norm_l2`,
`norm_max`, `kron`, `matpow` (exponentiation by squaring over `matmul_into`),
finite-`f32` `matrix_rank`/`matpow`/`det` over strided rank-2 operands, and
device-resident Cholesky, LU and QR with the solve/determinant/inverse forms
each factorization supports.

The 295-line release checklist that stood here is a per-PR record of that
surface; the release note itself belongs to `CHANGELOG.md`, and every entry is
recoverable from `git log -p -- checklist.md`. Residual distinctions worth
keeping: WGPU rank uses row-reduction pivots where Leto uses the SVD spectrum,
and WGPU determinant uses exact pivots with no tolerance where Leto uses its CPU
determinant algorithm.

## 0.7.0 reduction block-width dispatch [minor]
- [x] Added `reduction_with_width` so reduction WGSL generation, pipeline
  cache keying, intermediate output sizing, and dispatch group counts use a
  caller-selected power-of-two `BlockWidth`.
- [x] Kept `reduction` as the default-width API by delegating to
  `reduction_with_width(..., BlockWidth::DEFAULT)`.
- [x] Added contract coverage for width 128 integer reduction and
  non-power-of-two width rejection.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (29
  passed); `cargo test --doc --workspace --locked`; `cargo doc --workspace
  --no-deps --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  278,195 ns/iter; caller-owned 55,390 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Deeper gates attempted: `cargo
  semver-checks --workspace --all-features` blocked because the crates are not
  published in the registry; `cargo llvm-cov --workspace --locked` blocked by
  missing `llvm-tools-preview`. Evidence tier: typed API contract,
  value-semantic tests, and empirical benchmark.

## Codex review_gpu

- [ ] [HEPH-WGPU-DEVICE-REQUEST-FAULT](backlog.md#heph-wgpu-device-request-fault): collect PR #285 delivery and compact the completed item.
