# Backlog — hephaestus

<a id="hephaestus-scan-leto-001"></a>

## HEPHAESTUS-SCAN-LETO-001 — Share the Leto scan clause across scalar types — review
- Outcome: One scalar-generic long-line Leto differential validates every backend's scan path.
- Scope: conformance clause, exports, composite caller, five backend scan tests and this item; no kernel changes.
- Acceptance: Replace the type-specific helper with one generic clause, instantiate it for i32 and f32 on each backend, preserve signed inputs and exact Leto equality, and keep type_suffixed_fns at or below baseline.
- Depends on: [Atlas pin reconciliation ATLAS-GITLINK-DRIFT-056](../../backlog.md#ATLAS-GITLINK-DRIFT-056).
- Integrator: /root; regions: crates/hephaestus-conformance/src/scan.rs, src/lib.rs, src/assert_backend.rs, backend tests/scan_contracts.rs, backlog.md.
- Verification: fmt; host nextest 58/58; CUDA scan 1/1 on RTX 5080; WGPU contracts 8/8 with a required adapter; strict Clippy for conformance, host, CUDA, and WGPU; doctests; conformance reports zero regressions. Atlas overlay required unlocked local Cargo runs; standalone lock resolution remains covered by PR CI.
- Last-update: 2026-09-22.

<a id="heph-cuda-elementwise-scalar-declarations"></a>

## HEPH-CUDA-ELEMENTWISE-SCALAR-DECLARATIONS — Compile admitted elementwise scalars [patch] — todo
- Outcome: CUDA elementwise operations compile every scalar admitted by their bounds and preserve each operation's native scalar contract.
- Scope: existing CUDA binary/unary/scalar generators and shared tests; enumerate dependent scalar-bound generators before changing declaration composition. No dense-product scope expansion or widened arithmetic.
- Evidence: source-only review at `28769e1` plus product correction finds `elementwise/binary.rs::shader_source` and `elementwise/scalar.rs::shader_source` emit `T::TYPE_TOKEN` without its required declarations, although their bounds admit F16/Bf16. No elementwise device reproduction has run.
- Acceptance: reproduce missing declarations with real required-device operations; migrate generators to the shared dialect declaration owner; exact/derived-bound value and PTX oracles cover admitted types and reject unsupported native arithmetic.
- Dependencies: [scalar declaration owner](#heph-cuda-dense-product-scalars); priority P1; risk: dispatch failure. Authority: repository Change through merge; integrator assigned on claim.
- Verification: strict affected Clippy, bounded required-device scalar matrix, generated instruction review, docs and independent review.
- Last-update: 2026-09-08.

<a id="heph-cuda-product-address-range"></a>

## HEPH-CUDA-PRODUCT-ADDRESS-RANGE — Preserve representable product addresses [patch] — todo
- Outcome: rank-2 CUDA matrix and Kronecker products address validated layouts without signed 32-bit truncation or multiplication overflow.
- Scope: product metadata conversion and address generation, shared product tests and [ADR 0044](docs/adr/0044-device-neutral-dense-product-seam.md); no unrelated kernels.
- Evidence: source-only review of `28769e1` plus scalar-correction diff: `linalg/mod.rs::map_layout` admits u32 origins and i32 strides, while `matmul.rs` and `kron.rs` cast origins and coordinate products to signed int. No large-address device reproduction has run.
- Acceptance: reproduce with bounded metadata/source oracles at origins >= 2^31 and products beyond i32; derive checked signed address bounds, preserve negative strides, verify generated arithmetic and ordinary real-device strided products. Never execute a known invalid address natively.
- Dependencies: scalar compiler correction; priority P1; risk: device addressing correctness. Authority: repository Change through merge; integrator assigned on claim.
- Verification: focused metadata/codegen and required-device product tests, strict affected Clippy, docs and independent review; retain explicit physical-memory coverage limits.
- Last-update: 2026-09-08.

<a id="heph-cuda-product-empty-sum"></a>

## HEPH-CUDA-PRODUCT-EMPTY-SUM — Write zero for empty matrix sums [patch] — todo
- Outcome: matrix products with zero shared extent write the additive identity to every logical output cell while preserving storage outside the output view.
- Scope: CUDA matrix and batched products, allocating forms and shared product tests; no unrelated zero-size operations.
- Evidence: source-only review of `28769e1` plus scalar-correction diff: `linalg/matmul.rs::matmul_into` and `batched_matmul_into` return success when the shared extent is zero even with nonempty output; allocating forms therefore leave their output unwritten. No device reproduction has run.
- Acceptance: first reproduce `[m,0] * [0,n]` and batched equivalents with cloned nonzero sentinel outputs; assert exact zeros, untouched strided backing cells and unchanged inputs. Empty output remains a no-op; allocating forms return initialized zeros.
- Dependencies: scalar compiler correction; priority P1; risk: incorrect numeric output. Authority: repository Change through merge; integrator assigned on claim.
- Verification: all admitted scalar instantiations, dense/strided outputs and batches under required-device nextest budgets, strict affected Clippy, [ADR 0044](docs/adr/0044-device-neutral-dense-product-seam.md) contract update and independent review.
- Last-update: 2026-09-08.

<a id="heph-lockfile-queue-latency"></a>

## HEPH-LOCKFILE-QUEUE-LATENCY: Bound hosted guard queue latency [patch] - todo
- Outcome: the lockfile guard receives a hosted runner within its five-minute runtime target.
- Scope: shared guard scheduling and duplicate trigger load; preserve the guard and current required verification.
- Acceptance: record runner-assignment delay below five minutes on the next comparable PR/default-branch deliveries; exact locked guard still passes.
- Dependency: Atlas shared `.github/workflows/lockfile-guard.yml`, pinned here at `886d85b0`; diagnose account hosted capacity before changing routing.
- Evidence: PR #285 job `101619278287` queues 24m09s, then passes in 19s; earlier identical jobs queue 9m30s and 23m51s, then pass in 18s and 15s.
- Limits: `ubuntu-latest` works for other jobs; no configuration fault, billing failure, or platform-wide outage is established.
- Verification: `python scripts/lockfile.py --check` passes at `7a3a9ec` with 49 first-party Git sources; local execution preserves the guard while hosted delivery is delayed.
- Last-update: 2026-09-07; driver: [PR #285](https://github.com/ryancinsight/hephaestus/pull/285).

<a id="heph-nvrtc-loader-errors"></a>

## HEPH-NVRTC-LOADER-ERRORS — Preserve runtime compiler loader faults [patch] — todo

- **Outcome/scope:** preserve library, directory-enumeration and missing-export errors in `infrastructure/compiler.rs` runtime compiler acquisition; no driver ABI or mathematical kernel change.
- **Evidence:** `NvrtcDriver::get` discards required-symbol errors with `.ok()?`; `find_nvrtc_library` suppresses library and directory errors, reporting every failure as unavailable.
- **Acceptance:** real absent-library, present-invalid-library and missing-export cases retain their identities and error categories; existing real-device kernel compilation and dispatch contracts pass.
- **Dependencies/authority:** the native driver boundary item; authorized provider change, separate compiler acquisition contract.
- **Verification:** locked warning-denied checks, required-device bounded Nextest, native-loader failure tests and independent review; no fake compiler or weakened runtime budgets.

## HEPH-STAGGERED-3D-ROCM — ROCm kernels for the staggered pair [minor] — todo <a id="heph-staggered-3d-rocm"></a>

- **Outcome:** `RocmStaggered3DOps` implementing `Staggered3DOps<RocmDevice>`,
  judged by the same conformance clauses the WGPU and CUDA backends pass.
- **Consolidation requirement, not a copy:** the staggered kernel source is
  plain CUDA/HIP C with no vendor intrinsics, so the ROCm implementation would
  be byte-identical to the CUDA one. It is the second consumer, which is where
  consolidation fires: the source moves to a shared home both backends compile
  — not pasted into a second crate. Choosing that home is part of this item;
  the existing per-backend Laplacian sources are the precedent to avoid, not to
  follow.
- **Blocker:** no AMD device on the development host, so the differential
  against the CPU pair — the only oracle that makes the gathered transpose
  trustworthy — cannot run here. Landing an unverifiable copy of a kernel whose
  wall closure was hand-derived is what the CUDA increment's mutation check
  exists to prevent.
- **Re-open trigger:** an AMD device reachable from a development host or a CI
  runner, or a ROCm emulation path the conformance clauses can drive.
- **Last-update:** 2026-09-06.

## HEPH-CUDA-LAUNCH-DRAIN-REEVAL — original item (superseded 2026-09-01)

- Owner: unclaimed.
- Outcome: remove — or re-justify against current evidence — the Windows
  per-launch `cuCtxSynchronize` drain in
  `crates/hephaestus-cuda/src/application/pipeline.rs:250-260`.
- Evidence (audit 2026-08-27): the drain's justification cites managed-memory
  faults, but the backend now allocates only `cuMemAlloc_v2`
  (infrastructure/device.rs `alloc_bytes`, :381) and the managed path was
  removed (KS-8 closure). The drain serializes every launch and blocks CU-P1
  stream overlap (KS-7).
- Scope: the pipeline launch path only; allocation strategy is
  HEPH-CUDA-STREAM-ORDERED-ALLOC's concern.
- Acceptance: drain removed with kernel correctness and stress tests green on
  a Windows CUDA host, or a re-derived justification recorded at the drain
  site; either way the decision cites the run evidence.
- Dependencies: a Windows host with a runtime CUDA device — the current
  development machine compile-gates CUDA only, so removal cannot be validated
  here.

## HEPH-CUDA-STREAM-ORDERED-ALLOC — original item (superseded 2026-09-01)

- Owner: unclaimed.
- Outcome: device-buffer pooling or stream-ordered allocation for the CUDA
  backend so per-op alloc/free traffic stops serializing the device.
- Evidence (audit 2026-08-27): every op allocates `cuMemAlloc_v2` and frees
  `cuMemFree_v2` fresh (application/elementwise/unary.rs:99;
  application/reduction.rs:92, :121 — a fresh buffer per reduction pass);
  each free is an implicit device-wide synchronization.
- Direction: `cuMemAllocAsync`/`cuMemFreeAsync` on the legacy stream, or a
  sharded pool mirroring the wgpu backend's.
- Constraint (soundness): free-is-implicit-sync is currently the argument for
  dropping buffers referenced by in-flight kernels
  (application/pipeline.rs:206-213 with infrastructure/buffer.rs:82-98). Any
  move to async frees must first make that invariant API-contractual
  (event-ordered frees), not incidental.
- Acceptance: alloc/free no longer device-wide syncs on hot paths, drop
  soundness argument recorded at the free site, differential and stress tests
  green on a CUDA host.

## ✅ HEPH-WGPU-QR-DEVICE-Q [minor] [perf]: Device-side Q accumulation

- **Delivered**: PR #238 (`1fca9d6`, merged as `6a15a8b`).
  `GpuQrDecomposition::accumulate_q` builds Q on the device from the stored
  Householder reflectors against a `linalg::device_identity`, and the Python
  `qr` arm returns that buffer — `inner().q()` is no longer uploaded on the
  WGPU route. Transfer is `4mn + 8·min(m, n)` in place of `4m²`.
- **Verified 2026-08-31**: `accumulate_q` present at `qr.rs:137`; the Python
  WGPU arm calls it (`decomposition.rs:334`) with no `inner().q()` upload left
  on that route (the CUDA arm still uploads, as scoped); the contract case at
  `tests/contract.rs:3998` runs at both routing regimes. Lease released.
- **Integrator**: Claude session 5050c72a; lease: none.

- Superseded planning detail, retained for the audit trail:
- Lease: `crates/hephaestus-wgpu/src/application/decomposition/qr.rs`, the
  `qr` arm of `crates/hephaestus-python/src/decomposition.rs`, the QR
  contract cases, one visibility change in
  `crates/hephaestus-wgpu/src/application/linalg.rs`, and this item block.
- **Upstream enabler:** the reflectors live in `leto_ops::QrDecomposition`'s
  `pub(super)` `packed`/`heads`/`betas`, unreachable from this repo, and the
  delegating route returns leto's own decomposition — so read accessors
  belong in leto (upstream ownership), not a hephaestus-local copy of the
  factor. Filed and delivered as a leto [minor] API addition; this item
  consumes it after the repin.
- **Design:** `Q` is accumulated lazily, never during factorisation — an
  R-only caller (least squares) must not pay `O(m^2 n)` for a Q it discards.
  Start from a device identity (reusing `linalg::device_identity` rather
  than a second identity kernel), upload the packed factor plus per-reflector
  `(head, beta)`, and apply the reflectors in reverse (`Q = H_1(H_2(...H_k I))`)
  with one workgroup per Q column, mirroring the existing panel kernel's
  reduce-then-update shape. Transfer becomes `4mn + 8·min(m,n)` in place of
  the `4m^2` Q upload, and the `O(m^2 n)` accumulation moves off the host.
- Acceptance unchanged from the filing: differential against `inner().q()`
  within a tolerance derived from Householder backward stability, at both
  routing regimes, with transfer-count evidence.

- Outcome: the Python `qr` binding stops uploading `inner().q()`, because a
  device-resident **Q** exists to return.
- Evidence (2026-08-30): `crates/hephaestus-python/src/decomposition.rs`
  uploads `inner().q()` (`m`x`m` f32) on every WGPU `qr` call;
  `GpuQrDecomposition` exposes no device Q at all. This is the surviving
  half of `HEPH-WGPU-QR-DEVICE-FACTORS`, whose R half was retired by the
  premise correction recorded there.
- Scope: accumulate **Q** from the stored Householder reflectors on the
  device. The blocked path already owns a reflector-application kernel
  (`hephaestus-qr-hh-update`, applying panels to the trailing matrix), so
  the design question is whether Q can be built by applying those panels to
  an identity rather than a new kernel family. **Non-goals:** changing the
  factorization itself or the CUDA arm (no device on this host).
- Acceptance: Q accumulated on-device, differentially verified against
  `inner().q()` within a tolerance derived from Householder backward
  stability, with transfer-count evidence for the removed `4m^2` upload.
- Risk / change class: [minor] [perf]; a new device-side accumulation is
  numerically load-bearing and needs the differential oracle above before
  it can replace the upload.
- **Delivered 2026-08-31 (draft PR, merge-blocked):**
  `GpuQrDecomposition::accumulate_q` builds **Q** from a `linalg::device_identity`
  by applying the stored reflectors in reverse, one workgroup per column,
  with a 256-way tree reduction per dot product. A zero β is deliberately not
  skipped: β is a storage read, so branching on it around the workgroup
  barriers would be non-uniform control flow, and with β = 0 the update is
  already the identity. The Python `qr` arm calls it instead of uploading
  `inner().q()`, and reports shapes from `decomp.shape()` (`Q` is `m`x`m`,
  `R` is `m`x`n` by construction) so neither host factor is materialised.
- **Transfer:** `4m^2` uploaded becomes `4mn + 8·min(m, n)` uploaded — at the
  `[138, 129]` fixture 74.4 KiB becomes 70.5 KiB. QR requires `m ≥ n`, so
  `4mn ≤ 4m^2` always, with the win scaling as `m/n` and the two converging
  at `m = n` (square costs `8m` bytes more). The unconditional gain is the
  `O(m^2 n)` host accumulation itself, which no longer runs per call.
- **Differential evidence:** `qr_accumulated_q_matches_host_reference` runs at
  both routing regimes — `[70, 35]` (2 panels, delegating) and `[138, 129]`
  (5 panels, device schedule), each guarded against its intended route — and
  compares elementwise against `inner().q()` within `2·m·min(m, n)·ε`. The
  bound sums both accumulations' backward-stability error: `‖Q̂ − Q‖ ≤
  c(m,n)·ε` with `c(m,n) ≤ m·min(m,n)` (Higham ch. 19), reflectors being
  orthogonal so rounding is transported rather than amplified, and the
  device's tree reduction bounded below the host's sequential sum. Measured
  `max|Q_gpu − Q_host| = 5.96e-8` at both shapes (one ulp at unit magnitude)
  against tolerances `5.84e-4` and `4.24e-3`. Orthogonality is asserted
  independently at `[138, 129]` — `max|QᵀQ − I| = 9.51e-7` against the same
  bound — because an elementwise check against a same-order reference cannot
  see a reflector order that is wrong in both.
- **Liveness proved:** with the shader's reflector iteration reversed to
  forward order, the case fails at `Q[0, 1]` for `[70, 35]` — device
  `-9.99e-4` against host `1.002e-3`, delta `2.00e-3` exceeding the
  `5.84e-4` tolerance — and it is the only case that fails. Restored and
  re-run green; the mutation is not committed.
- **Gates:** `cargo nextest run -p hephaestus-wgpu` 31/31, 0 skipped, against
  a real adapter (172 contract cases); warning-denied all-target Clippy over
  `hephaestus-wgpu` and `hephaestus-python`; `cargo test --doc` 2/2 and
  `cargo fmt --check` clean. No wall-clock claim.
- **Merge blocker:** the committed `Cargo.lock` pins a leto revision without
  `packed()`/`heads()`/`betas()`, so this builds locally only through the
  Atlas overlay and CI cannot pass until leto PR #134 merges and hephaestus
  repins leto. PR stays draft until then; re-open trigger: the leto repin.

## HEPH-CUDA-OXIDE-MEMCPY2D-ABI [patch] — in-progress

- Incorporated into [HEPH-CUDA-DRIVER-BOUNDARY](#heph-cuda-driver-boundary); provider-owned ABI replaces the upstream dependency and per-row workaround.

## HEPH-FFT-PROVIDER-1 [minor] [arch] [perf] — in progress

- Owner: Codex session `01a0253c-6013-7552-99cc-36bbbcf77f6d`; provider
  readback correction is on `perf/wgpu-readback-completion-pool` and consumer
  closure is in Apollo/Kwavers.
- Lease: none. The retained-parameter candidate on
  `feat/wgpu-bound-parameters` adds a device-provenance-checked mutable
  uniform update while preserving fixed bindings and launch geometry.
  Warning-denied all-target Clippy, WGPU Nextest (28/28 in 31.381 seconds),
  Rustdoc, rustfmt, and diff checks pass. `cargo-semver-checks` did not reach
  API comparison because its local baseline clone failed on an oversized
  packed-object entry. Independent exact-candidate review of `e6218da` is
  GREEN; exact-lock hosted gates and merge remain.
  Readback PR #232 merged as `b4e170e` from exact reviewed head
  `cf0907e`. Provider PR #230 merged as `48bb731`; device-preflight PR #231
  merged as `1636301`.
  The candidate replaces the per-readback completion channel with eight fixed
  slots acquired before submission. Reader/callback ownership quarantines
  pending state through poll errors, callback delay/cancellation, and unwind;
  capacity overflow allocates before submission. Deterministic tests hold all
  retained slots, force capacity plus one, distinguish terminal outcomes, and
  prove bounded release/reuse. A one-slot Loom model checks every interleaving
  of concurrent reader/callback release, callback completion or cancellation,
  and racing reacquisition (1/1 in 0.671 seconds). Warning-denied all-target
  Clippy, exact-candidate WGPU nextest (28/28 in 25.096 seconds), doctests (2/2),
  and rustdoc pass. Independent exact-head review is GREEN; hosted lockfile and
  host verification pass. CUDA, ROCm, Metal, and WGPU hosted checks were still
  running when the PR merged and remain a collection watchpoint. Apollo and
  Kwavers consumer closure remains.
- Outcome: Hephaestus becomes the single accelerator owner of dense complex FFT
  execution, exposes one prepared device-neutral contract for ranks one through
  three, and provides the WGPU implementation needed by Kwavers. Kwavers then
  selects `Leto` (Apollo CPU FFT over Leto arrays) or `Hephaestus` at the
  operation boundary without fallback.
- Scope: provider-neutral split-complex operands, shape/direction/normalization
  validation, prepared caller-owned device dispatch, WGPU radix and Bluestein
  execution, 1-D/2-D/3-D conformance, warm-allocation and device-residency
  evidence, and the dependent Apollo/Kwavers cutover that deletes their
  superseded WGPU FFT implementations.
- Non-goals: moving FFT arithmetic into Leto; a Leto-to-Hephaestus dependency;
  hidden accelerator-to-host fallback; real-to-complex packing; vendor-specific
  consumer APIs; or a performance claim before matched end-to-end evidence.
- Acceptance: ADR 0053 is accepted; the core contract validates ranks 1..=3,
  nonzero checked shapes, dense split-complex storage, non-aliasing components,
  and fixed forward/inverse normalization before mutation; prepared WGPU plans
  execute power-of-two and non-power-of-two axes with no allocation, pipeline
  compilation, host transfer, or capability probe in repeated dispatch; one
  generic conformance suite covers 1-D/2-D/3-D analytical spectra, Apollo/Leto
  differential results, inverse round trips, invalid shapes/layouts/aliases,
  and unchanged outputs on preparation rejection; Kwavers exposes closed
  `Leto`/`Hephaestus` selection and its PSTD path uses Hephaestus; Apollo and
  Kwavers retain no consumer-owned WGPU FFT shader or plan after cutover.
- Verification plan: warning-denied core/WGPU checks, configured Nextest,
  doctests, SemVer checks for the additive provider surface, real-device WGPU
  conformance, allocation instrumentation after plan preparation, codegen/source
  residue scans, matched rank/shape benchmarks, independent architecture review,
  and exact-head hosted provider plus Kwavers consumer CI.
- Dependencies: Kwavers PR #663 already delivers closed `Leto | Hephaestus`
  selection and direct Hephaestus WGPU execution. Apollo WGPU deletion waits
  for native-f16 provider parity; CUDA remains a later provider slice because
  Apollo's CUDA execution is not yet redundant.
- Local evidence: the core FFT planner and operation seam are warning-clean;
  configured Nextest passes 106/106 `hephaestus-core` tests in 0.683 seconds,
  including ranks one through three plus rank/layout/alias/address rejection;
  warning-denied core rustdoc passes. Commands ran standalone against the
  committed lockfile to avoid the Atlas development overlay rewriting Git
  sources as local paths. The WGPU implementation now runs one rank-generic
  axis plan for ranks one through three, prebinding pipelines, immutable
  parameter buffers, bind groups, and dispatch grids at preparation. Direct-DFT
  and inverse-round-trip device tests pass for radix and Bluestein shapes at all
  three ranks; a sparse impulse at prime length 262,147 checks selected bins
  against the closed form after range-reduced Bluestein phase preparation.
  Prepared forward and inverse plans encode into one provider-neutral command
  stream, cross-device dispatch is rejected, and the planner rejects dispatch
  grids beyond the acquired device's actual workgroup limit. All 10 FFT tests
  pass in an exact post-review 3.404 seconds; the full required-device WGPU
  package passes 230/230 with no skips in 61.075 seconds before the final
  composition-test strengthening. Warning-denied all-target Clippy and rustdoc
  pass. Core and WGPU SemVer checks each pass all
  196 applicable minor checks against the clean `b9ace296` baseline. The
  required-device bounded benchmark smoke passes 1D 1,024, 1D 1,000 Bluestein,
  1D 65,536, 2D 256x256, 3D 64x64x64, and 3D 32x32x33 Bluestein. The benchmark
  calls the provider-neutral composition seam, checks independent forward DFT
  bins under a depth-derived `gamma(k)` bound, and bounds submission, readback,
  and CI process time. Independent blocker-only re-review is clean after
  correction of benchmark-oracle, bounded-wait, provider-neutral composition,
  and fallible host-allocation findings.
  A consumer integration audit then exposed two warm-path costs hidden by the
  first prepared shape: cloned operand handles now make plans independently
  storable, pack/unpack binds those fixed buffers directly, and the two duplicate
  full-volume allocations plus four per-transform copies are gone. The exact
  static reductions are `8N` prepared bytes and `16N` warm-copy bytes; 11/11
  focused and 231/231 full real-device WGPU tests pass, including dropped
  source handles and in-pass consumer composition. Warning-denied all-target
  WGPU Clippy and rustdoc pass; the minor-policy SemVer check passes 196/196.
  Provider PR #222 merged as `cfadc373`. The pre-cutover Kwavers WGPU PSTD
  baseline is 10.09 ms/step for 50 steps on a 256x128x128 lossless grid. The
  initial matched Hephaestus workload took 13.810 ms median for the six transform
  pairs, falsifying direct staged-plan cutover. A device-qualified fused radix
  strategy reduces the same workload to 7.9974 ms median (42.1%), one dispatch
  per active axis, and no full-volume workspace; at this shape the forward and
  inverse plans remove 64 MiB of workspace and retain only two 4 KiB root tables.
  Direct-oracle rank tests and singleton-axis/no-workspace tests pass. The full
  required-device WGPU package passes 233/233 with no skips in 62.109 seconds;
  this preserves coverage but confirms the separately tracked 26-binary device
  acquisition/topology defect. Complete step timing remains the consumer cutover
  gate before the old shader is deleted. The current increment instantiates this
  same rank-generic plan, strategy, and WGSL family for `f32` and native
  `eunomia::F16`; binary16 rejects a device without `ShaderF16` before plan
  allocation or operand mutation and has no host or wider-scalar fallback. One
  generic real-device conformance body covers both scalars across ranks one
  through three, fused/staged radix, Bluestein, singleton axes, analytical bins,
  and inverse round trips. The first independent review found wider shader-side
  trigonometry, weak scale-relative assertions, and incomplete rank/special-value
  coverage. The correction precomputes roots and reciprocal scales directly in
  the selected scalar, uses normwise relative oracles that reject all-zero
  output, and adds both scalar widths across ranks one through three, nontrivial
  staged Fourier modes, and NaN/infinity/zero cases. Those tests exposed and
  fixed staged twiddle angle halving, `u32` index-product overflow, and a
  radix-four second-half-circle lookup. Focused FFT Nextest now passes 22/22 in
  17.081 seconds, and the complete bounded benchmark smoke passes every 1-D,
  2-D, 3-D, Bluestein, PSTD, and scalar-width case in 9.198 seconds. The scalar
  benchmark validates sampled forward bins against an independent direct DFT
  before inverse dispatch and proves that identity output fails the oracle.
  Warning-denied core/WGPU all-target Clippy and Windows AArch64 compilation,
  doctests, rustdoc, and core/WGPU SemVer checks (196/196 each) pass.
  Independent architecture/performance review is green. The canonical
  standalone lock resolves under `--locked` with 33 first-party Git sources.
  Linux AArch64 execution is not claimed: the local
  cross build stops in the `alloca` build script because
  `aarch64-linux-gnu-gcc` is absent, before project code is compiled. Grouped
  warm-dispatch tests preserve prepared command-slice and workspace-buffer
  identities; source inspection establishes no Hephaestus-owned allocation,
  pipeline or bind-group construction, transfer, or copy on that path, but does
  not observe opaque driver allocations. On an RTX 5080 through Vulkan, paired
  forward/inverse Criterion regression-slope time estimates (95% confidence
  intervals) are 162.71 microseconds [161.25, 164.67] for binary32 and 171.31
  [168.20, 174.41] for binary16 at 65,536 elements; at 64 cubed they are
  221.64 [219.22, 225.42] and 215.52 [212.69, 218.68] microseconds. The result
  does not support a universal
  binary16 speed claim. The selected-axis correction validates and executes
  only a nonempty unique in-range axis set, normalizes inverse transforms by
  active extents, and preserves the all-axis convenience. Direct-DFT and inverse
  row oracles cover `[3, 8]` and `[3, 5]`; invalid selections reject before
  mutation. Retained grouped binding moves consumer pipeline, uniform, bind
  group, and dispatch-grid preparation out of warm encoding while preserving
  fixed-resource and per-encode sequence-device provenance. Parameter size is
  checked against enabled limits, and scoped WGPU failures preserve causal
  precedence: allocation, internal, validation, then host binding. Focused
  real-device Nextest passes 14/14 in 15.486 seconds; warning-denied all-target
  Clippy, formatting, doctests, rustdoc, core/WGPU SemVer (196/196 each), and
  two independent static reviews are green at `90572d3`. Apollo/Leto
  differential coverage, consumer deletion, and exact-head hosted verification
  remain open.

## HEPH-BOOK-REGROUND-1 [patch] [docs] — in progress

- Owner: current Atlas session on `fix/hephaestus-book-reground-1`; the dirty
  primary checkout remains outside this scope.
- Outcome: every `docs/book/` chapter describes the API this repository ships,
  and the book gate cannot pass over a fabricated chapter.
- Scope: `docs/book/{compute_device,capabilities,device_buffer,elementwise_ops,`
  `dense_reductions,decomposition_seam,wgpu_backend,cuda_rocm,stack_position}.md`,
  the affected executable examples, and this provider PM record.
- Non-goals: new chapters for uncovered seams, Rust source changes, or
  `SUMMARY.md` restructuring.
- Acceptance: every named identifier resolves in the source or a named
  dependency; executable fences compile or are explicitly `rust,no_run` for
  device-only paths; `mdbook test docs/book` passes; and the Pages workflow
  runs that gate.
- Verification plan: exact-source API audit, `cargo fmt --all -- --check`,
  locked package build, `mdbook build`, strict link check, and hosted exact-head
  book verification. The shared Atlas target may make local mdBook dependency
  discovery non-representative; hosted clean-run evidence remains authoritative.
- Local evidence: the affected chapters are source-rewritten; formatting,
  warning-denied `hephaestus-host` Clippy, locked package build, host Nextest
  `2/2`, mdBook HTML build, and `mdbook-linkcheck2 --standalone` pass. Local
  `mdbook test` reaches the executable chapters but cannot be counted because
  the shared Windows target contains multiple historical `hephaestus-core` and
  `themis` rlibs and does not stage Windows proc-macro DLLs; the clean Linux
  hosted job remains the acceptance oracle. The parallel `host_backend.md`
  cleanup is intentionally outside this lane.

## HEPH-BOOK-TEST-2026-08-20 [patch] — in progress

- PR #214 exposed a real mdBook 0.5.4 contract defect after the package build
  passed: the included HostDevice and capabilities examples lacked explicit
  extern crate declarations for their staged provider crates.
- The bounded fix adds the declarations to both included examples and repins
  the shared Atlas workflow to hash-preserving staging revision 20c9398.
- Local formatting, mdBook build, strict links, and diff checks pass. Hosted
  exact-head rerun remains required; local locked Cargo verification remains
  subject to the shared Atlas overlay lock-form mismatch.

## HEPH-FDTD-PROVIDER-1 [minor] [arch] — in progress

- Owner: Codex on `codex/hephaestus-fdtd-107`; scope: the device-neutral
  collocated 3D FDTD contract, WGPU implementation, provider value contract,
  and owner-keyed PM/ADR records. Kwavers consumer cutover is a dependent
  increment.
- Outcome: make Hephaestus the single owner of typed FDTD buffers, geometry,
  stencil dispatch, and provider execution ordering.
- Non-goals: CPU solver ownership, source injection, medium construction,
  consumer comparison policy, CUDA/ROCm kernels, or runtime performance claims.
- Acceptance: validated f32 `Fdtd3dParams`, `FdtdMedium`, and `FdtdVelocity`
  types; one `Fdtd3dOps` seam; WGPU velocity-then-pressure dispatch with
  spacing-aware central differences; invalid-storage rejection; and a
  provider-versus-independent-one-step contract test. Hosted provider CI and
  the Kwavers integration sweep remain required for closure.
- Verification: local formatting, core/WGPU compilation, focused Nextest, and
  a device-required hosted WGPU contract run at the exact implementation head.

## HEPH-MOIRAI-PACKAGE-1 [patch] — in progress

- Owner: Codex `/root`; scope: root dependency package identities and registry
  versions, clean lockfile, focused WGPU resolution gate, and release records.
- Acceptance: the `moirai` Rust import resolves package `moirai-runtime` from
  Moirai's default branch without compatibility code, and all six publishable
  packages pass exact-source crates.io dry runs.
- Status: the exact external graph resolves Moirai `b7988419`, Leto
  `a5d53ca9`, and the published package identity; format and the focused locked
  WGPU package check pass. Exact-head CI exposed stale Mnemosyne patch keys;
  those keys now match the published package identities. The first crates.io
  dry run then rejected unversioned normal Git dependencies before upload; all
  normal path and Git dependencies now carry their registry versions. Exact-
  source dry runs, hosted verification, and merge remain.

## HEPH-STATEFUL-ZERO-LR-1 [patch] — in-progress

- Owner: Codex on `codex/hephaestus-zero-learning-rate`; scope: stateful-update
  parameter validation, focused contracts, and release records.
- Outcome: every stateful-update parameter contract accepts a finite zero
  learning rate while retaining strict positive epsilon and finite-domain
  checks.
- Acceptance: all five rules construct at zero learning rate, negative and
  non-finite rates remain rejected, focused Nextest and warning-denied Clippy
  pass, and exact-head hosted checks pass before merge.
- Status: implementation and focused local verification complete; hosted
  verification pending.

- Composition note (2026-07-31 late, session-2026-07-30-board-ssot): the
  sparse-seam commit on this lane also carries the rocm-pivot frontier's
  in-flight snapshot (board files + wgpu linalg identity-contract work) —
  an over-broad `git add -A` from the shared tree. Nothing altered or
  lost; the sparse-seam content is disjoint.

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: [Atlas ADR 0001](../../docs/adr/0001-gpu-accelerator-substrate.md).

## HEPH-STATEFUL-UPDATE-1 [minor] [arch] — in-progress

- Owner: Codex on `codex/hephaestus-stateful-update`; scope: ADR 0045,
  provider-neutral stateful update vocabulary and validation, WGPU/CUDA/ROCm/
  Metal implementations, shared conformance, and synchronized release records.
- Outcome: expose one provider-owned, monomorphized stateful elementwise seam
  capable of SGD, Adam, RMSProp, AdamW, and AdaGrad without consumer-authored
  accelerator formulas or host execution.
- Non-goals: Coeus consumer cutover, optimizer API compatibility shims,
  reduced-precision admission, benchmark-instrument changes, or unmeasured
  runtime and memory claims.
- Acceptance: all five rules execute through one generic request/planner;
  validation precedes mutation and rejects invalid shape/span/layout/alias/
  hyperparameter contracts with typed errors; every backend instantiates the
  shared Leto-differential suite; no provider silently executes on another
  backend; warning-denied focused gates and exact-head provider CI pass.
- Risk/change class: `[minor] [arch]`; additive public provider seam. Coeus's
  later fallible cutover is a separate breaking consumer increment.
- Status: done 2026-08-01. Core planner and all four provider dispatch
  implementations pass warning-denied focused gates. Direct Leto differential
  conformance passes on physical WGPU and CUDA devices; ROCm's adapterless
  contract passes. Independent source review approves the provider ownership,
  Metal substrate boundary, and pre-mutation validation. Implementation head
  `fc0605c` passes WGPU `30713623614`, CUDA `30713623627`, ROCm `30713623615`,
  and native macOS Metal `30713623612`; the docs-only closeout head must pass
  the same matrix before merge. The Coeus consumer cutover is the next item.

## HEPH-ATTENTION-PROVIDER-1 [minor] [arch] — provider side complete; Coeus cutover open

- Composition note (2026-07-31, session-2026-07-30-board-ssot): commit
  `a23ee9b` on this lane bundles the attention frontier's uncommitted
  snapshot (56 files; it compiled and the cuda suite passed 142/142 at that
  revision) together with a disjoint cuda strided-meta refactor that was
  being committed from the shared tree at the moment the tree switched to
  this branch. The refactor is extracted to master as `fdb6980`; identical
  content merges cleanly when this lane lands. Attention work is otherwise
  untouched.
- Owner: Codex on `codex/hephaestus-attention-provider`; scope:
  provider-owned scaled dot-product attention forward and additive backward
  across WGPU, CUDA, ROCm, and Metal, shared Leto-differential conformance,
  ADR 0040, and direct Coeus integration.
- Outcome: accelerator attention dispatch remains device-resident and routes
  through one backend-neutral, monomorphized Hephaestus seam; CPU semantics
  remain owned by Leto.
- Non-goals: consumer-authored kernels, host execution, silent provider
  fallback, compatibility adapters, and performance claims without matched
  measurements.
- Acceptance: unmasked, causal, broadcast keep-mask, fully masked, strided,
  forward, and independently selected additive-gradient cases agree with Leto;
  validation and preparation failures are mutation-free; Coeus removes local
  attention kernels and routes CPU to Leto and accelerators to Hephaestus; all
  affected warning-denied, Nextest, doctest, SemVer, and exact-head CI gates
  pass.
- Risk/change class: `[minor] [arch]`; additive provider contract and direct
  consumer cutover under ADR 0040.
- Status: provider implementation and exact-head hosted gates complete at source
  `702eba8`, merged provider default `4714b8c` on 2026-08-17. The shared
  attention contract was structurally cleaned by moving its download assertion
  into `src/attention/assertions.rs`; the provider conformance scan returns
  `oversized_files=38`, down from 39. CUDA `32026666522`, ROCm `32026666500`,
  WGPU `32026666544`, and Metal `32026666549` pass. The direct Coeus cutover
  remains the open dependent item.

## HEPH-PARAMETERIZED-UNARY-1 [minor] [arch] — in-progress

- Owner: Codex on `codex/hephaestus-parameterized-unary`; scope:
  provider-owned runtime-parameter unary expressions, Hardtanh and Threshold
  forward/gradient implementations across WGPU, CUDA, ROCm, and Metal,
  shared Leto-differential conformance, ADR 0032, and direct Coeus integration.
- Outcome: runtime activation parameters remain data supplied at dispatch, not
  consumer-authored shader source, and every backend computes the same
  parameter-sensitive values and kink conventions as Leto CPU.
- Non-goals: unrelated activation families, consumer compatibility paths,
  silent CPU fallback, and performance claims without matched measurements.
- Acceptance: non-default parameter and boundary cases pass one generic
  conformance suite on each available provider; Coeus removes its local
  Hardtanh/Threshold expressions and routes directly through Hephaestus; all
  affected warning-denied, nextest, doctest, and exact-head CI gates pass.
- Risk/change class: `[minor] [arch]`; additive provider contract and direct
  consumer cutover under revised ADR 0032.
- Status: in-progress 2026-07-31.

## HEPH-CONVOLUTION-PROVIDER-1 [minor] [arch] — in progress

- Owner: Codex on `codex/hephaestus-compute-seams`; claimed scope:
  `crates/hephaestus-core/src/domain/convolution/`,
  backend `application/convolution/` modules, focused provider contracts,
  ADR 0039, and this item. The scope also absorbs the stale WGPU elementwise,
  scan, and full-reduction seam increment required to keep the shared backend
  exports warning-clean and device-safe.
- Outcome: one fallible, monomorphized accelerator convolution seam with
  regular and transposed forward/additive-backward implementations for WGPU,
  CUDA, ROCm, and Metal, using Leto parameters as the SSOT.
- Non-goals: Coeus caller migration, release/version transitions, dynamic
  dispatch, host fallback, and runtime or memory claims without matched
  measurements.
- Acceptance: ranks 1 through 3 validate and dispatch through borrowed strided
  device views; each backend differentially matches Leto for supported scalars;
  invalid shape/storage/alias/address contracts fail before mutation; selected
  provider failures return typed errors without host transfer or provider
  change; focused package gates and exact-head provider CI pass.
- Risk/change class: `[arch] [minor]`; additive provider surface with
  cross-backend kernel, address-width, and device-failure risk.
- Status: done 2026-07-30; delivered through PR #159. Leto
  regular/transposed forward/backward ownership is
  merged through parameter-SSOT promotion at `f896c43`; WGPU, CUDA, ROCm, and
  Coeus closure audits identify the exact kernels, fallbacks, missing ranks,
  and safety contracts to migrate. The core seam and shared planner now pass
  package check, seven focused value-semantic Nextest contracts, and doctests.
  CUDA passes the shared rank-one through rank-three `f32`/`f64`
  regular/transposed forward/backward conformance matrix on physical hardware
  in 4.6 seconds. WGPU passes the shared rank-one through rank-three `f32`
  matrix in 0.8 seconds. Its combined convolution and compute-seam lane passes
  18/18 tests in 8.4 seconds, including invalid shader compilation, arbitrary
  writable overlap, device identity, and foreign-buffer rejection before
  mutation. ROCm now owns native HIP kernels and passes its adapterless
  feature-off build; feature-enabled Linux compilation and device execution
  remain hosted gates. Writable plans reject arbitrary overlapping strided
  layouts, and backend address checks cover convolution projection
  intermediates before dispatch. Warning-denied Clippy passes for
  core/conformance/WGPU/Metal, CUDA with its native feature, and ROCm's
  feature-off configuration under one coherent rustup toolchain and the shared
  target directory. Exact-head hosted CI passes the CUDA, ROCm, WGPU, and macOS
  Metal feature and adapterless contract jobs; hardware-only NVIDIA and AMD
  jobs skip because their repository variables are unset.

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

## Open

The KS-1 … KS-10 kernel-seam program this section tracked is delivered: the core
dialect and op vocabulary, the authored-kernel seam in core, per-backend
impls of the core op vocabulary, `KernelDevice`/`CommandStream` for WGPU and
CUDA, the grouped authored-kernel seam, the device-capability and
acquisition-policy vocabularies, the order-preserving tiled scan, CUDA driver
handle serialization and memoization, the CUDA stream/pinned-staging batch, the
managed-memory WDDM launch drain, and the `hephaestus-metal` retention decision
all landed, and each has its own record elsewhere on the board. The 43-entry
status ledger that stood here is program history; recover any entry with
`git log -p -- backlog.md`.

**Live residuals this section was the only home of:**

- **CU-P1 (async stream pipelining/overlap)** -- the narrower staging half is
  closed; what remains is custom per-device `CUstream`s for compute/transfer
  overlap, which is lower-value on this crate's primary target
  (Windows/WDDM, where KS-8 already forces a `cuCtxSynchronize` drain after
  every kernel launch). Reassess scope before starting.
- **`MetalDevice`/`MetalBuffer` retirement is tracked as `ATLAS-ARCH-011`** and
  is blocked on `ATLAS-SUBSTRATE-002` (the `coeus-metal` consumer), not on
  anything in hephaestus: the removal was executed here, verified green, and
  reverted solely for that consumer. The forwarding layer makes zero native
  Metal API calls across 5,449 lines.
- **CUDA and Python semver rustdoc is blocked** by a `cargo-semver-checks`
  limitation, not by this crate.

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

<a id="heph-kernel-tail-accuracy"></a>

## HEPH-KERNEL-TAIL-ACCURACY — Correct cancellation, overflow and tail underflow in operator renderings [patch] — todo
- Outcome: every operator rendering listed in [ADR 0061](docs/adr/0061-operator-value-semantics.md) Decision 6 stays within a tolerance derived from its chosen form's published relative bound against the operator's value function, on Vulkan, DX12 and Metal (WGSL), CUDA C and HIP C.
- Scope: `crates/hephaestus-core/src/domain/ops.rs` and `parameterized.rs` renderings of Expm1, Log1p, Elu, Celu, Softplus, Mish, MishGrad, Erfc (WGSL), Gelu, GeluGrad, GeluTanh, GeluTanhGrad, Silu, SiluGrad, and the tests pinning their strings; no value-function or seam change.
- Evidence: ADR 0061 reviews two to five measured the defects and the rejected candidate forms in f32 against f64 (recorded in Decision 6); no device run yet.
- Acceptance: a per-operator sweep over the failing ranges on each available backend; tolerances cite the chosen form's source; the WGSL `log` accuracy bound is confirmed against the specification's accuracy table before any form relies on it; Metal is measured wherever a macOS host is available, since its default math mode may fold the cancellation-free forms.
- Dependencies: none for the renderings; the host clauses exercising these operators wait on it (ADR 0061 Consequences). Priority P1; risk: silent wrong values at tails.
- Verification: strict Clippy, device sweeps on the RTX 5080 (Vulkan, DX12, CUDA), and on Metal where a macOS host is available (otherwise the Metal gap is recorded as residual risk), doc sync, independent review.
- Last-update: 2026-09-18.

<a id="heph-ray-integral-direction-contract"></a>

## HEPH-RAY-INTEGRAL-DIRECTION-CONTRACT — Integrate rays in world length for any direction [patch] — todo
- Outcome: `RayIntegralOps` states whether the integral covers the whole line or the ray from its origin, and returns `∫ field dl` in world units for a direction of any non-zero length.
- Scope: `RayIntegralOps` docs in `crates/hephaestus-core/src/domain/volume.rs`, the wgpu, CUDA, HIP and Metal kernels, the host implementor, and the conformance clause.
- Evidence: every kernel weights samples by the chord length in the ray parameter `t` (`acc * actual`), which is world length only when `|direction| = 1`; a direction `2·u` halves the result. `t_enter` may be negative, so a ray starting inside the volume also integrates behind its origin. The clause uses unit directions and outside origins only, so neither behaviour is pinned. Source reading at `origin/master` 2026-09-18; no device run.
- Acceptance: the contract names the integration domain; a clause case with a non-unit direction and one with an origin inside the volume, both against analytical oracles, pass on every backend and the host.
- Dependencies: none. Priority P2; risk: silently scaled integrals for callers passing unnormalized directions.
- Verification: strict Clippy, device clause runs (RTX 5080: Vulkan, DX12, CUDA), host nextest, doc sync.
- Last-update: 2026-09-18.
