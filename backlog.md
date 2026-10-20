# Backlog — hephaestus

<a id="heph-eunomia-010-cohort"></a>
## HEPH-EUNOMIA-010-COHORT — restore provider compatibility [minor] — blocked

- Priority: correctness.
- Outcome: Hephaestus 0.22 consumes one Eunomia 0.10/Leto 0.46 provider identity without Rust implementation changes.
- Scope: workspace/internal and provider versions, canonical lock, pre-1.0 migration and backend CI's exact source graph. Existing macOS/kernel work retains its owner.
- Acceptance: strict workspace Clippy, complete host and available Metal value/rejection tests, docs/doctests, API comparisons and applicable hosted backend CI on the exact candidate.
- Dependency: Moirai #623 and Leto #363 are merged at `44a962f0` and `8b4ab407`; the refreshed lock selects all eight provider defaults. Preserve master #494 (`b91180aa`) as a merge parent. All 204 registry records and every Rust source remain identical to that base.
- Integration record: [PR #493](https://github.com/ryancinsight/hephaestus/pull/493) owns final-head native, portable/Linux-SDK API, hosted backend, fresh-consumer and publication evidence. Merge follows all applicable checks and Leto 0.46 registry receipts; automatic publication is authorized. Manual NVIDIA/AMD hardware execution remains unverified.
- Blocker/receipts (folded from the retired checklist): record the final-head fresh-consumer run, hosted acceptance, and the six 0.22.0 registry receipts on PR #493; the macOS verification checklist (docs/verification/macos/checklist.md) retains its hardware and kernel scope.

<a id="heph-macos-metal-conformance-1"></a>

## HEPH-MACOS-METAL-CONFORMANCE-1 — Canonical Metal provider verification [patch] — todo
- Priority: correctness.
- Remaining acceptance: exact-head hosted Metal and required provider gates,
  followed by the authorized provider merge.
- Outcome: macOS CI triggers on WGPU, core, conformance, facade, and test-policy
  changes; it runs minimal/default WGPU and existing Metal contracts with Metal
  acquisition and both required-device flags. Existing public APIs remain stable.
- Evidence: 2026-10-13, aarch64 macOS, Rust 1.95.0, standalone `--locked` graph:
  151 default WGPU/Metal, 85 minimal, 88 decomposition-only, 85 sparse-only,
  and 341 core/conformance/host Nextest tests pass with zero skips; strict
  Clippy, feature/facade checks, four doctests, and warning-denied rustdoc pass.
  All 191/142/189/147 consolidated cases retain their value or rejection
  contracts. Unsupported FP64 executes typed rejection before mutation for
  fusion, stateful updates, and cross-entropy backward. The shared local
  Hermes/Hephaestus consumer reduces 515 exact dyadic inputs to -37.625 on
  both native NEON and Metal under Rust 1.97.0.
- Design: ADR 0047 makes Metal an adapter of WGPU. FFT, fusion, triangular
  masking, cross product, arg-reduction/top-k, interpolation, and adaptive
  pooling therefore use the existing WGPU implementations on Metal.
- Publication closure: partition upstream decomposition aggregation, WGPU
  device ownership/lifetimes, and shared CTC dispatch into leaf modules, and
  replace the old utility modules with identity/tolerance and storage-binding
  responsibilities. Integrate upstream `cf811ae9`, partition its typed
  comparison/arithmetic dispatch and host decomposition type/solve utilities,
  and remove the internal upload-only forwarding function. Mathematical operation order remains unchanged; later
  conversion closure makes representation boundaries explicit.
  Implementation-bearing manifests fall from 17 to
  the recorded 14; utility modules fall from two to zero. See the scoped
  [design note](docs/verification/macos/design.md).
- Upstream conversion closure: replace 64 moved casts without exclusions;
  preserve pointer identity, validated grids, native canary inputs, and Python
  integers. Reject oversized CUDA count words before dispatch. Three new
  bit-pattern/reference/boundary tests raise the complete required-Metal
  workspace run to 780 passing tests; strict Clippy also passes.
- Hosted calibration closure: replace the empirical precision cuts with
  operation/precision error envelopes derived from the portable builtin model;
  preserve every probe value, reported error, and live grade assertion. Exact
  rational tests prove outward bounds; zero identities and required Metal
  acquisition reject invalid results and device faults.
- Python closure: remove the deprecated workspace-wide extension feature that
  prevents embedded test executables from linking Python. Preserve Maturin's
  environment-selected extension mode and use PyO3's cdylib build helper. Both
  Rust tests pass; the WGPU lifecycle executes its three binding clauses, and
  a real imported extension computes exact add/sum/min/max Metal values without
  linking the interpreter library. External dependency identities stay fixed.
- Loader closure: recognize dyld's direct-image missing-path grammar without
  concealing dependency, invalid-image, or missing-symbol faults. Driver/NVRTC
  symbol tests select the real macOS system library. Nine loader contracts
  pass, and required CUDA still rejects absent hardware. The complete locked
  Rust 1.97.0 workspace passes 777 tests with zero Nextest skips.
- Conversion closure: replace bare casts throughout fixed/staggered FD
  conformance, CTC and window metadata, host CTC/FD, prepared-output identity
  tests, and Metal Laplacian fixtures with checked/lossless/native-precision
  conversions. Conformance/core/host/ROCm/WGPU counts fall from
  50/48/37/169/319 to 30/24/19/162/298 under the unchanged budget. Fixture
  domains, values, tolerances, and operation order retain their contracts.
- Residual: hosted workflow execution and Intel hardware are unmeasured;
  the separate forwarding-crate retirement and native-kernel/performance
  roadmap retain their existing scope. Target version: 0.20.4.

<a id="heph-shape-ops-provider-1"></a>

## HEPH-SHAPE-OPS-PROVIDER-1 — Device-neutral shape/indexing seam [minor] [arch] — todo
- Priority: feature.
- Status: split one op at a time (per ADR 0062's precedent), not one
  `ShapeOps` seam. Delivered: `PadOps` (ADR 0062), `RollOps` (ADR 0069),
  `TriangularOps` (ADR 0070, tril/triu) — all host + wgpu.
- Triangular residual: CUDA and ROCm implementors plus the
  `assert_backend_contract` aggregate fold remain tracked scope; host + wgpu
  delivery does not close those backend obligations. The canonical WGPU
  implementation also passes the required-device Metal lane (ADR 0047).
- Outcome: cat, split, stack, gather, scatter, index_select, index_put,
  masked_fill, tile, where, sort, and nonzero remain, implemented
  natively on every accelerator backend (no host round-trip), split into
  their own items as each is taken up.
- Scope: per-op core domain seam + wgpu/cuda/rocm application modules;
  Metal uses WGPU per ADR 0047. Conformance clauses compare against Leto per op.
- Driver: coeus's accelerator bridge has no path for this family today and
  falls back to host execution for every one of these ops (relayed consumer
  audit, 2026-09-26).
- Dependencies: upstream ownership (standards: backend hierarchies) — lands
  here first, generic accelerator layer plus per-vendor device impls only;
  coeus binds afterward.
- Verification: per-op conformance clause across backends, boundary/empty
  cases, strict Clippy, independent review.
- Last-update: 2026-10-12.

<a id="heph-topk-argminmax-provider-1"></a>

## HEPH-TOPK-ARGMINMAX-PROVIDER-1 — CUDA/ROCm for topk/argmax/argmin [minor] — todo
- Priority: feature.
- Status: `ArgReduceOps` (argmax/argmin, ADR 0063) and `TopKOps` (topk, ADR
  0066) both delivered — host + wgpu, differential-tested including tie-break
  cases. Narrowed to the remaining backends and the `assert_backend_contract`
  aggregate fold (deferred until a seam has every provider implementation).
- Outcome: `ArgReduceOps`/`TopKOps` implemented for cuda/rocm. The canonical
  WGPU implementations pass on Metal under HEPH-MACOS-METAL-CONFORMANCE-1.
- Scope: per-backend implementors following the existing wgpu kernels'
  algorithms (host/wgpu already agree on tie-break by construction — matching
  algorithm, not just contract); conformance already covers the shared
  clauses, so this item is implementation-only.
- Driver: coeus consumer audit, 2026-09-26.
- Verification: existing differential + boundary (ties) tests per backend,
  strict Clippy, independent review.
- Last-update: 2026-09-26.

<a id="heph-embedding-gather-provider-1"></a>

## HEPH-EMBEDDING-GATHER-PROVIDER-1 — Embedding-gather gradient (scatter-add) [minor] — todo
- Priority: feature.
- Status: `EmbeddingOps::scatter_add_into` delivered per ADR 0087 — host +
  wgpu (two-pass range-check + output-parallel accumulate), ascending
  position-order contract with an order-sensitive conformance fixture, plus
  rejection clauses incl. the empty-table edge. Narrowed to the cuda/rocm
  implementors and the `assert_backend_contract` aggregate fold (deferred
  until a seam has every provider implementation).
- Outcome: `scatter_add_into` implemented for cuda/rocm. The canonical WGPU
  implementation passes on Metal under HEPH-MACOS-METAL-CONFORMANCE-1.
- Scope: per-backend implementors following the existing wgpu kernels'
  algorithms (ascending position order, matching algorithm, not just
  contract); conformance already covers the shared clauses, so this item is
  implementation-only.
- Driver: coeus consumer audit — needed for the embedding-lookup layer's
  training path.
- Verification: existing differential + repeated-index accumulation (incl.
  the order-sensitive fixture) + out-of-range rejection tests per backend,
  strict Clippy, independent review.
- Last-update: 2026-10-20.

<a id="heph-interpolation-provider-1"></a>

## HEPH-INTERPOLATION-PROVIDER-1 — CUDA/ROCm for interpolation [minor] — todo
- Priority: feature.
- Status: `InterpolationOps` (nearest/linear, ADR 0067) delivered — host +
  wgpu, align-corners convention, differential-tested including the
  `in_len == 1` and `out_len == 1` degenerate cases. Bilinear is not a
  separate mode: on a regular grid it composes as two calls to
  `interpolate_axis_into` (documented in the ADR), so it needs no distinct
  binding. Narrowed to the remaining backends and the
  `assert_backend_contract` aggregate fold (deferred until a seam has every
  provider implementation).
- Outcome: `InterpolationOps` implemented for cuda/rocm. The canonical WGPU
  implementation passes on Metal under HEPH-MACOS-METAL-CONFORMANCE-1.
- Scope: per-backend implementors following the existing wgpu kernel's
  algorithm (host/wgpu already agree on the align-corners mapping and the
  nearest-mode tie rule by construction); conformance already covers the
  shared clause, so this item is implementation-only.
- Driver: coeus consumer audit, 2026-09-26.
- Verification: existing differential + boundary (degenerate extents) tests
  per backend, strict Clippy, independent review.
- Last-update: 2026-09-27.

<a id="heph-adaptive-pooling-provider-1"></a>

## HEPH-ADAPTIVE-POOLING-PROVIDER-1 — CUDA/ROCm for adaptive pooling [minor] — todo
- Priority: feature.
- Status: `AdaptivePoolingOps` (avg/max, ADR 0068) delivered — host + wgpu,
  differential-tested including the non-divisible-extent case (windows can
  overlap, matching the reference algorithm) and the `out_len == 1`/
  `out_len == in_len` degenerate cases. 2D pooling is not a separate mode:
  it composes as two calls to `adaptive_pool_axis_into` (documented in the
  ADR). Narrowed to the remaining backends and the `assert_backend_contract`
  aggregate fold (deferred until a seam has every provider
  implementations).
- Outcome: `AdaptivePoolingOps` implemented for cuda/rocm. The canonical WGPU
  implementation passes on Metal under HEPH-MACOS-METAL-CONFORMANCE-1.
- Scope: per-backend implementors following the existing wgpu kernel's
  window-derivation formula; conformance already covers the shared clause,
  so this item is implementation-only.
- Driver: coeus consumer audit, 2026-09-26.
- Verification: existing differential + boundary (non-divisible extents)
  tests per backend, strict Clippy, independent review.
- Last-update: 2026-09-27.

<a id="heph-device-dot-cross-provider-1"></a>

## HEPH-DEVICE-DOT-CROSS-PROVIDER-1 — Device-resident cross product; coeus dot-binding gap [patch] — todo
- Priority: feature.
- Status: audited (ADR 0064). `DenseVectorOps::dot`/`norm_l2`/`norm_l1`/
  `norm_max` already dispatch entirely on-device on wgpu (only the reduced
  scalar downloads) — the reported "coeus copies full operands to host"
  symptom for `dot` is a coeus-side integration gap (not calling the
  existing seam), not a hephaestus provider gap; report back to coeus rather
  than duplicate. `CrossProductOps<D, T>` (host + wgpu) delivered for the
  half that had no seam at all.
- Outcome: cross-product batched over flat `(x, y, z)` triples, on-device.
- Scope: `hephaestus-core`/`hephaestus-host`/`hephaestus-wgpu` delivered;
  cuda/rocm implementors remain, plus the `assert_backend_contract` aggregate
  fold once every provider implements it. The canonical WGPU implementation
  also passes on Metal under HEPH-MACOS-METAL-CONFORMANCE-1.
- Driver: coeus consumer audit, 2026-09-26.
- Verification: differential vs the standard basis identities (host +
  real-device wgpu), strict Clippy, independent review.
- Last-update: 2026-09-26.

<a id="heph-fused-expr-rocm-metal-1"></a>

## HEPH-FUSED-EXPR-ROCM-METAL-1 — Fused-expression dispatch on ROCm [minor] [arch] — todo
- Priority: feature.
- Status: `RocmFusionOps` landed; the C-dialect generator, scalar
  vocabulary, layout record, and pipeline key are now owned once in
  `hephaestus_core::fusion` (ADR 0055 rev. 2026-10-20) with CUDA/HIP
  byte-identity pinned by tests, and the per-backend twins deleted.
  Remaining: the canonical WGPU expression dispatch passes on Metal under
  HEPH-MACOS-METAL-CONFORMANCE-1.
- Outcome: Metal leg of fusion conformance verified.
- Scope: execution evidence only; no further fusion code is planned here.
- Driver: coeus consumer audit — coeus cannot rely on fusion outside
  wgpu/cuda today.
- Verification: Metal fusion contracts green in hosted backend CI, strict
  Clippy, independent review.
- Last-update: 2026-10-20.

<a id="heph-rocm-windows-hip-runtime-1"></a>

## HEPH-ROCM-WINDOWS-HIP-RUNTIME-1 — Support the validated Windows HIP substrate [minor] — todo
- Priority: feature.
- Outcome: split the current Linux-only ROCm runtime boundary so the Windows
  HIP SDK path is enabled only after its runtime compiler, device discovery,
  buffer, stream, and convolution/attention contracts pass on supported AMD
  hardware.
- Scope: `crates/hephaestus-rocm` infrastructure, pipeline, and device
  contracts; retain the Linux path's semantics and typed unavailable-device
  errors until the Windows path is verified.
- Evidence: AMD documents Windows HIP SDK support but calls it a subset of
  ROCm; this crate currently rejects every non-Linux `rocm` build in `lib.rs`.
- Verification: supported Windows HIP device conformance for attention and
  convolution, strict Clippy, source-generated-kernel review, and independent review.

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

<a id="heph-moirai-package-1"></a>

## HEPH-MOIRAI-PACKAGE-1 [patch] — blocked

- Owner: Codex `/root`; scope: root dependency package identities and registry
  versions, clean lockfile, focused WGPU resolution gate, and release records.
- Acceptance: the `moirai` Rust import resolves package `moirai-runtime` from
  Moirai's default branch without compatibility code, and all six publishable
  packages pass exact-source crates.io dry runs.
- Blocked by: the moirai-0.9 provider advance (moirai-sync `^0.8.2` no longer
  resolves against Moirai HEAD 0.9.2); this branch carries that advance.
  Re-open if 0.9 resolution regresses.

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
- Status: 2026-10-14 — the #427 rescue branch carries a 17-file ULP-sweep
  oracle system (`crates/hephaestus-conformance/src/accuracy/`, new) plus
  four backend sweep drivers; ported verbatim it compiles on master, and
  11/16 sweeps fail as true kernel findings, not oracle defects: missing
  ±inf guards for 8 forms (gelu, gelu-grad, gelu-tanh, gelu-tanh-grad,
  silu, silu-grad, mish, mish-grad — WGSL evaluates `-inf * 0` = NaN
  where the defining value is ∓0); erfc systematic divergence (971/4363
  over bound, worst 25336 ULP); log1p 9.52 vs 6 (possible driver `log`
  accuracy); celu 11M ULP at x=-1e-8, PARTIALLY cured by #484 (GPU now
  renders `expm1`; re-sweep to confirm, do not assume). Next step: port
  the oracle system first, then harden kernels until 16/16 pass under
  this item's derived-bounds discipline (difficulty never justifies
  narrowing).
- Last-update: 2026-10-14.

<a id="heph-wgsl-f64-transcendentals-1"></a>

## HEPH-WGSL-F64-TRANSCENDENTALS-1 — Width-valid f64 transcendentals on Vulkan [minor] — todo
- Priority: feature.
- Outcome: f64 `exp`/`log`/`pow`/`tanh` and the other GLSL.std.450
  precision-restricted builtins compute on Vulkan through WGSL routines built
  only from width-unrestricted f64 operations (add, multiply, `fma`,
  `floor`, `ldexp`), lifting the ADR 0072 refusals op by op.
- Scope: `crates/hephaestus-wgpu/src/application/canary.rs` and the WGSL
  composition each gated seam uses; `docs/adr/0072-*`.
- Acceptance: each routine documents its range reduction and a derived
  relative-error bound; the canary probe grades it Exact (`<= 1e-12`) on
  lavapipe and NVIDIA Vulkan, with special values (NaN, +-Inf, overflow and
  underflow edges) asserted against CPU truth.
- Needs: none. Next step: derive the `exp` reduction and its bound.

<a id="heph-retained-reductions-1"></a>

## HEPH-RETAINED-REDUCTIONS-1 — Retained-reduction providers on all backends [minor] — todo
- Outcome: every backend implements `RetainedReductions` (retain dense
  map-reduction outputs for reuse) with a shared conformance oracle.
- Scope: `crates/hephaestus-core/src/domain/vector.rs` (trait exists);
  five backend `vector`/`dense_vector` files (trailing impl blocks
  omitted from #482); new
  `crates/hephaestus-conformance/src/retained_reduction/` oracle; five
  backend `retained_reduction_contracts.rs` tests;
  `into_retained`/`retain_dense_*`/`dispatch_with` in map-reduction files.
- Evidence: ~587 lines of new backend behavior on
  `rescue/green-hephaestus-20261012` (reference implementation; port, do
  not redesign). Deferred from the #427 selective rescue: the CUDA/ROCm/
  WGSL paths need a GPU executor (CI is adapterless there; a local CUDA
  device exists for the CUDA leg).
- Acceptance: oracle contracts pass per backend on available hardware;
  adapterless legs compile and skip cleanly.
- Dependencies: none. Priority P2.
- Verification: shared oracle per backend, strict Clippy, doc sync,
  independent review.
- Last-update: 2026-10-14.

<a id="heph-rocm-product-validation-1"></a>

## HEPH-ROCM-PRODUCT-VALIDATION-1 — Mirror CUDA product validation on ROCm [patch] — todo
- Outcome: ROCm matrix and Kronecker products get the checked-address and
  empty-sum contracts #487 landed for CUDA.
- Scope: `crates/hephaestus-rocm/src/application/linalg/` (`mod.rs`,
  `kron.rs`, `matmul/{single,batched}.rs`),
  `tests/dense_product_contracts.rs`.
- Evidence: #487 landed CUDA-only because the #427 rescue has no ROCm
  equivalent to port.
- Acceptance: `validate_signed_address_range` + the 9-case oracle ported;
  kron/single call sites wired (verify the HIP batched kernel widens to
  `long long` before excluding it); k=0 dispatches-and-zeroes (verify
  zero-tile behavior of the HIP kernel first); empty-product device
  matrix mirrored. No ROCm hardware locally or in CI: land behind the
  existing compile gates + adapterless legs; do not land blind device
  claims.
- Dependencies: none. Priority P1; risk: device addressing correctness.
- Verification: focused metadata/codegen and adapterless product tests,
  strict affected Clippy, docs and independent review.
- Last-update: 2026-10-14.

<a id="heph-roll-f64-gpu-1"></a>

## HEPH-ROLL-F64-GPU-1 — Run the n-D roll contract at f64 on CUDA/ROCm [patch] — todo
- Outcome: the #483 n-D roll contract runs at f64 on CUDA and ROCm devices.
- Scope: `crates/hephaestus-{cuda,rocm}/tests/roll_contracts.rs` (one line
  each).
- Evidence: #483 runs i32+f32 on GPU; `CudaRollOps`/`RocmRollOps` are
  generic over `DialectScalar` so f64 compiles, but no device was
  available to verify dispatch.
- Acceptance: add the f64 lines when a CUDA/ROCm device is at hand and
  watch them pass; do not land blind.
- Dependencies: none. Priority P3.
- Verification: required-device contract run.
- Last-update: 2026-10-14.

<a id="heph-backend-contract-fold-1"></a>

## HEPH-BACKEND-CONTRACT-FOLD-1 — Fold all seams into assert_backend_contract [minor] — todo
- Outcome: every device-neutral seam folds into the
  `assert_backend_contract` aggregate with per-backend instantiation.
- Scope: `crates/hephaestus-conformance/src/assert_backend.rs`, five
  backend aggregate instantiations.
- Evidence: the #427 rescue's 37-seam fold cannot compile today (6
  backend×seam pairs missing: cuda/rocm/metal × Embedding/Roll — roll
  landed in #483, embedding is the named blocker). The rescue
  `assert_backend.rs` is the reference; re-derive, do not copy (it also
  de-genericizes some parameters).
- Acceptance: `BackendUnderTest` extended with the missing seams;
  aggregate instantiates per backend with all seams bound.
- Dependencies: [HEPH-EMBEDDING-GATHER-PROVIDER-1](#heph-embedding-gather-provider-1).
  Priority P3.
- Verification: aggregate contract suite per backend, strict Clippy,
  independent review.
- Last-update: 2026-10-14.

<a id="heph-cuda-reduce-throughput-1"></a>

## HEPH-CUDA-REDUCE-THROUGHPUT-1 — Fix CUDA axis-reduction serial collapse [patch] — todo
- Outcome: CUDA reductions run at memory-bandwidth pace.
- Scope: `hephaestus-cuda` axis-reduction kernel + plan (the Coeus entry path),
  then the Coeus full-reduction seam. `coeus-cuda` delegates identically to
  `coeus-wgpu` (verified by reading), so the defect is below Coeus.
- Evidence: caller-measured 2026-10-14 (RTX 5080, driver 610.60) via the
  Coeus `ops_bench` reduction-crossover group (f32 scalar sum end to end,
  scalar readback included): CUDA device path 96us at n=256 rising
  linearly to 13.5ms at n=1048576 (~3.4ns/B, about 300MB/s on a ~1TB/s
  part); the same workload on WGPU runs 89us to 0.79ms (17x faster at
  1M). Host staging beats the CUDA device path at every measured size.
- Root cause (proven 2026-10-15, same host): `coeus_ops::sum` reshapes to
  [N] and calls `sum_axis(axis=0)`; the seam ranks to [1,N] and dispatches
  `axis_reduction_kernel`, which spawns ONE THREAD PER OUTPUT that loops
  the whole axis serially (`reduction/shaders.rs`). At N=1M one thread
  executes 1M dependent FP adds. Direct hephaestus-cuda probe (no Coeus):
  13.39/13.55ms steady-state at [1,1M] — reproduces the 13.5ms exactly.
  The parallel full reduction on the same box does 1M in ~120us
  end-to-end (~20us kernel) but is unreachable from Coeus (no seam);
  launch path is async and NVRTC is cached (ruled out).
- Fix directions: (A) parallelize the axis kernel (intra-block tree +
  grid-stride over the axis; `plan_axis_reduction` is CUDA-local in
  practice — only hephaestus-cuda consumes it); (B) add a Coeus
  full-reduction seam and route scalar reductions through it (~100x on
  the measured workload). (A) fixes the class (long-axis multi-output
  collapses identically); (B) fixes the symptom fastest.
- Acceptance: root-cause fix landed; bandwidth-paced reduction at 1M+
  elements; the Coeus CUDA staging threshold (currently 1048576)
  re-measured and revisited.
- Dependencies: none. Priority P1; risk: every CUDA reduction pays 10-20x.
- Verification: device measurement before/after on the same host and
  workload, strict Clippy, independent review.
- Last-update: 2026-10-14.
