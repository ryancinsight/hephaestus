# Backlog — hephaestus

## HEPH-STAGGERED-3D-2026-09-04 — Device 3-D staggered gradient/divergence pair [minor] [arch] — review <a id="heph-staggered-3d-2026-09-04"></a>

- **Integrator:** Claude on `feat/hephaestus-staggered-3d`; **lease:**
  `crates/hephaestus-core/src/domain/staggered.rs`,
  `crates/hephaestus-wgpu/src/application/stencil/`,
  `crates/hephaestus-metal/src/application/stencil.rs`,
  `crates/hephaestus-conformance/src/staggered.rs`, `Cargo.lock` — 2026-09-04.
- **Outcome:** the device half of `leto_ops::StaggeredLeapfrog3D`.
  `Staggered3DOps<D>` with `Staggered3DParams` and `StaggeredAxis` in
  hephaestus-core, WGSL kernels in hephaestus-wgpu, Metal by delegation, and
  shared conformance clauses. Completes the CPU/GPU seam Coeus PR #369 opened:
  a consumer binds one trait and reaches either backend.
- **Separate trait, not new methods on `StencilOps`:** a backend without
  staggered kernels would otherwise have to supply bodies, and a body returning
  zeros or an error is a mock wearing a trait impl. CUDA and ROCm advertise the
  capability when they have it (`HEPH-STAGGERED-3D-CUDA-ROCM`).
- **The divergence gathers.** Leto scatters `-Gᵀ`, which makes the adjoint true
  by construction; a GPU cannot scatter without atomics, so the kernel gathers a
  transpose derived by hand — including the wall closure the CPU comment warns
  is the easy thing to get wrong. The derivation is written out in the module
  docs and is checked three ways rather than trusted.
- **Taps are a parameter, not derived in core.** hephaestus-core carries Leto
  layout vocabulary and no CPU compute dependency (atlas ADR 0001); a linear
  solve there would be exactly the dependency that boundary excludes. The caller
  passes `leto_ops::staggered_first_derivative_coefficients` output, and the
  conformance clauses close the gap that opens.
- **Documented capability difference:** the device kernels require
  `extent >= 2N` on the swept axis so one reflection step is exact.
  `Staggered3DParams::new` rejects thinner grids with a typed error; Leto's
  looping reflection still serves them on the CPU. A rejected configuration,
  not a silent divergence.
- **Evidence (2026-09-04):** `cargo fmt --check`, `cargo clippy --locked
  --workspace --all-targets -- -D warnings`, `cargo nextest run --locked -p
  hephaestus-core -p hephaestus-host -p hephaestus` **121/121**, `cargo test
  --locked --doc --workspace --exclude hephaestus-python`, `cargo doc --locked
  --workspace --no-deps` warning-free. The WGPU contract suite ran against a
  **live adapter** with `HEPHAESTUS_WGPU_REQUIRE_DEVICE=1`: **187/187** cases,
  up from 179, the eight new ones being the CPU differential on every axis at
  orders 2/4/6/8, the constant-field wall check, the device-side adjoint
  identity, the thin-grid rejection, the storage-length rejection, and the
  shared conformance clause.
- **The differential tests were proven live, not assumed:** flipping the sign of
  the low-wall reflected term in the gathered divergence failed exactly
  `staggered_divergence_matches_cpu_on_every_axis`,
  `staggered_high_order_matches_cpu`, and `the_device_pair_is_a_negative_adjoint`
  — the three that should catch it — and the mutation was reverted.
- **Decision record:** [ADR 0057](docs/adr/0057-device-staggered-pair.md).
- **Follow-on:** `HEPH-STAGGERED-3D-CUDA-ROCM` — CUDA and ROCm kernels for the
  same trait. The conformance clauses already exist and judge them on the same
  three oracles; until then those backends simply do not implement
  `Staggered3DOps`, which is the honest state and the reason it is a separate
  trait.
- **Last-update:** 2026-09-04.

## HEPH-PROVIDER-MERGED-2026-09-04 [patch] [arch] — review <a id="heph-provider-merged-2026-09-04"></a>

- **Integrator:** Codex on `build/hephaestus-source-identity`; **lease:** none.
- **Outcome:** remove obsolete Aequitas, Eunomia, Leto, and Moirai revision
  pins after merge so Hephaestus exports one layout and numeric type identity.
- **Acceptance:** standalone source scans and Apollo NUFFT compile use one
  Eunomia and Leto identity; WGPU-backed crates share the serialized test
  resource; configured Hephaestus gates pass; no consumer
  patch or conversion layer is introduced.
- **Dependencies:** Aequitas #51, Eunomia #87, Leto #168, and Moirai #256 are
  merged; **Last-update:** 2026-09-04.
- **Evidence:** source graph has one Eunomia and one Leto identity at Leto
  `3c1f9f1`; rejected Mnemosyne PR head `a07f999` is absent; all-target
  check and Clippy, format, focused WGPU/Python tests, doctests, and rustdoc
  pass. The full 435-test run reaches 406 passes before a serialized WGPU test
  exceeds the unchanged 60-second budget; each implicated test passes alone in
  0.4–3.3 seconds. Independent static judge: approve.

## HEPH-SEMVER-BUDGET-IDENTITY-2026-09-03 [patch] [arch] — done <a id="heph-semver-budget-identity-2026-09-03"></a>

- **Outcome:** Consume `KernelResourceBudget` through `moirai-gpu`, the
  planner facade, so fresh provider graphs cannot split the budget type across
  Mnemosyne source revisions.
- **Scope:** WGPU/CUDA planner call sites, their direct dependency manifests,
  CI/release workflow callers, Cargo.lock, ADR 0002, changelog, and this item.
  Moirai owns the public export in PR #256; no duplicated wrapper or conversion
  is permitted.
- **Acceptance:** Hephaestus constructs `moirai_gpu::KernelResourceBudget`,
  no direct `mnemosyne-memory-core` edge remains, standalone locked checks and
  warning-denied provider gates pass, and the hosted SemVer failure is removed.
- **Follow-up source edge:** closed by `HEPH-PROVIDER-MERGED-2026-09-04` after
  the corrected Leto and Moirai provider changes merged.
- **Risk / delivery:** [patch] internal dependency ownership. Merged as PR
  #270 at `7d0a474`; hosted SemVer, CUDA, ROCm, WGPU, Metal, host, lockfile,
  and documentation gates pass. The temporary Moirai pin was removed after PR
  #256 merged under `HEPH-PROVIDER-MERGED-2026-09-04`.

## HEPH-EUNOMIA-LAYOUT-SEAM-2026-09-03 [major] [arch] — done <a id="heph-eunomia-layout-seam-2026-09-03"></a>

- **Outcome:** Make Eunomia's native `Pod`/`Zeroable` markers and byte-cast
  functions the single first-party device-layout contract across Hephaestus.
- **Scope / non-goals:** Migrate `hephaestus-core`, the host reference device,
  all shipped backend implementations, conformance tests, and owned ABI
  metadata; remove direct first-party `bytemuck` imports and manifest edges.
  WGPU/CUDA/ROCm vendor internals and unavoidable transitive dependencies are
  not replaced by a local adapter or compatibility shim.
- **Acceptance:** `ComputeDevice`, `DeviceApi`, and every public operation seam
  bind `eunomia::Pod`; owned `repr(C)`/transparent metadata derives Eunomia
  markers; host↔device byte views use `eunomia::layout` with no copy added;
  all first-party crates compile and their value-semantic suites pass; direct
  source and manifest scans contain no Hephaestus-owned `bytemuck` contract.
- **Dependencies / risk:** The co-evolution pins used while Eunomia PR #87 and
  Aequitas PR #51 were open were removed after merge by
  `HEPH-PROVIDER-MERGED-2026-09-04`. This breaks the public generic bound and is
  a major co-evolution change; vendor transitive graphs remain outside this
  item.
- **Evidence:** all-target check, valid feature Clippy, format, lockfile check,
  default nextest (437/437), CUDA (165/165), WGPU (34/34), host/core (121/121),
  doctests, and warning-denied rustdoc pass. **Commit:** `c24de79`. **Last-update:** 2026-09-03.

## HEPH-FUSION-SEAM-2026-09-02 [minor] [arch] — done

- **Outcome:** Move runtime-rank expression fusion into the device-neutral seam and WGPU provider; [ADR 0055](docs/adr/0055-fusion-seam.md).
- **Scope / acceptance:** Core fusion contracts, borrowed dynamic views, provider-owned WGSL generation/cache/dispatch, layout and resource validation, and differential WGPU contracts; Coeus cleanup remains a follow-on migration.
- **Evidence:** After merging `origin/master` (including the mbind link fix), exact-source CUDA checks pass with and without `cuda,decomposition`; ROCm adapterless checks pass with and without `decomposition`; WGPU nextest `34/34`, workspace clippy, format, and locked WGPU all-target checks pass locally. The lockfile consumes merged Leto `070d52c`.
- **Delivery:** Merged [PR #265](https://github.com/ryancinsight/hephaestus/pull/265) as `eeee35b23cbbc074b8c89c8d9dd024480d1ae96c`.
- **Residual:** Post-merge hosted CUDA, ROCm, WGPU, host, docs, and lockfile checks were queued at collection; local exact-source gates are green, while hosted hardware evidence remains external.

## HEPH-CUDA-QR-DEVICE-Q-2026-09-01 [minor] [perf] — done

- **Outcome:** materialize the ordinary QR orthogonal factor **Q** on CUDA
  from the provider's compact Householder representation, then route the
  Python CUDA `qr` result through that device buffer.
- **Scope / non-goals:** CUDA ordinary QR Q accumulation, the shared CUDA
  identity kernel, the Python CUDA `qr` arm, focused contracts, ADR 0024, and
  Unreleased documentation. Factorization, least-squares, pivoted QR, and the
  already-device-resident **R** path remain unchanged; **Q** stays lazy so
  callers that do not request it allocate no `m²` output.
- **Acceptance oracle:** both CUDA QR entry routes produce a device **Q** that
  agrees with Leto's host accumulation within the Householder reduction bound;
  the real blocked route is independently orthogonal; empty shapes preserve
  identity semantics; a foreign dispatch device rejects before allocation or
  upload; the Python CUDA arm contains no `inner().q()` or host Q upload.
  The kernel launches one 256-thread block per Q column and reuses the existing
  device-identity implementation without a second identity kernel.
- **Performance model / stop condition:** the binding replaces a `4m²`-byte
  host Q upload plus host O(`m²n`) accumulation with `4mn + 8·min(m,n)` bytes
  of compact-factor uploads and device O(`m²n`) accumulation. No timing claim
  is made without a controlled benchmark; reject if value semantics regress or
  a least-squares-only call begins allocating Q.
- **Risk / delivery:** additive public CUDA parity is SemVer minor. Required
  evidence is focused debug/release CUDA values on a physical adapter,
  warning-denied CUDA/Python Clippy and Rustdoc, doctests, formatting, source
  routing audit, and SemVer analysis where the tool can complete.
- **Outcome / evidence:** source `b6c002a` adds lazy device-Q accumulation,
  transfers the retained R buffer without a device clone, and removes the
  Python CUDA host-Q path. RTX 5080 focused debug/release contracts pass 8/8
  (`e7ba55ed-934f-430a-851c-b7b961b32d99` and
  `404bf93f-459a-4a4d-826a-3c1a0b2fb0d5`); the CUDA Python binding contract
  passes 1/1 (`af00f59e-312d-400a-9634-b4cfd0b0d151`); exact-source
  CUDA/Python Nextest passes 165/165 in 35.870 s
  (`cfb70b91-4911-4d54-92ee-e90126a53f5c`). Warning-denied all-target,
  all-feature Clippy and Rustdoc, doctests, no-default CUDA checking,
  formatting, diff, locked-source, and routing audits pass. The public surface
  is additive; `cargo-semver-checks` remains uncollected because the baseline
  clone exceeds the local tool's entry-size limit. No throughput claim is made
  without a controlled timing instrument.
- **Ownership:** integrator=Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`;
  branch=`perf/cuda-qr-device-q`; lease=none; last-update=2026-09-01;
  remaining=independent review, PR, and merge collection.

- **Independent review (2026-09-01, Claude, RTX 5080 / CUDA 13.3 / driver 610.47):**
  merged as PR #246. `cargo nextest run --package hephaestus-cuda --features
  cuda,decomposition --locked` passes 161/161 in debug and 161/161 under
  `--cargo-profile release` on the physical adapter — the release-profile
  evidence the acceptance asked for and the earlier hardware run had not
  supplied. Source audit: the Python CUDA `qr` arm calls
  `decomp.accumulate_q(device)` and `into_r_buffer()`; the only `inner().q()`
  text remaining is a comment explaining what is *not* materialised. Reviewer
  did not author the change.
## HEPH-CUDA-HOST-ROUNDTRIP-SPLIT-2026-09-01 [minor] — done

- **Independent hardware evidence (2026-09-01):** `4bfbed8` is an ancestor of `1ffe9859`, the revision on which the RTX 5080 run of `cargo nextest run --package hephaestus-cuda --features cuda,decomposition --locked` passed 161/161 (see PR #246 review comment). The device-side split's contracts were therefore exercised on hardware by a reviewer who did not author them.

- **Context:** the wgpu and CUDA arms of the Python decomposition bindings are
  asymmetric. wgpu splits packed factors **on device** via
  `hephaestus_wgpu::split_packed_lu` (a shader,
  `crates/hephaestus-wgpu/src/application/decomposition/split.rs:98`). CUDA has
  no counterpart — `crates/hephaestus-cuda/src/application/decomposition/`
  holds lu/qr/cholesky/svd/schur/… but no `split.rs` — so the binding falls
  back to the host `split_packed_lu(&[f32], n)`
  (`hephaestus-core/src/domain/decomposition.rs:532`), downloading and
  re-uploading around it.
- **Evidence (three sites, `crates/hephaestus-python/src/decomposition.rs`):**
  - `:89-92` — `lu()`: downloads n² floats, splits on host, uploads 2n².
  - `:214-218` — same shape on the `lu_buffer()` path.
  - `:358-361` — QR uploads `q_host`/`r_host` from a **leto host**
    computation, so that arm may not be a device factorization at all.
- **Cost:** 3n² floats of PCIe traffic per `lu()` call that the wgpu arm does
  not pay — 12 MB at n=1024 — for a triangular mask that is pure indexing.
  Same defect class the Cholesky triangle-mask work closed (4n² → 4n), left
  open on the other backend.
- **Not a missing abstraction.** The `decomposition_seam` handles are already
  residency-correct: `factors()`/`r_buffer()`/`lower()` return
  `&D::Buffer<f32>`, and `solve(&self, device, rhs: &D::Buffer<f32>) ->
  Result<D::Buffer<f32>>` is device-in/device-out, so residency already
  survives chained operations. Only this backend's missing kernel forces the
  roundtrip. Do **not** scope this as a placement or residency layer.
- **Acceptance oracle:** a CUDA `split_packed_lu` matching the wgpu shader's
  contract, differential-tested against the host `split_packed_lu` on
  identical packed input — **exact equality**, since this is copy-and-mask
  with no arithmetic, so no tolerance applies; the three binding sites carry
  no `download_owned`/`upload` pair; `:358` either becomes a device path or
  its host computation is documented as deliberate with the reason.
- **Classification:** additive public CUDA parity is SemVer minor. At source
  `4bfbed8`, CUDA QR's retained **R** stays on-device while host **Q**
  accumulation remains explicit. Follow-up item
  `HEPH-CUDA-QR-DEVICE-Q-2026-09-01` retires that recorded remainder.
- **Narrowed 2026-09-01:** PR #246 (`feat(cuda): Accumulate QR Q on device`) closes the QR site (`:358-361`) on device with no host staging; independently verified here, 161/161 hardware contracts on the RTX 5080 at `1ffe9859`. Remaining scope is exactly the two LU packed-factor split sites (`:89-92`, `:214-218`), which need the CUDA `split_packed_lu` kernel. Nothing else changes.
- **Blocker check:** needs CUDA hardware. An RTX 5080 is present on this host
  — verify before deferring (recorded blockers expire). If the CUDA feature
  does not build here, that is the first finding, not grounds to close.
- **Outcome / evidence:** source `4bfbed8` adds the public CUDA split, routes
  both Python LU variants through it, and keeps CUDA QR **R** on-device. RTX
  5080 focused debug/release contracts pass 5/5 with exact host-oracle values;
  CUDA/Python Nextest passes 161/161 in 37.067 s; warning-denied all-target
  Clippy and Rustdoc, doctest, formatting, diff, locked source, and routing
  audits pass. The routing audit finds two CUDA LU splits, one device-local QR
  **R** clone, and zero forbidden host roundtrip pairs. `cargo-semver-checks`
  did not reach comparison because its baseline clone failed with `Entry too
  large to fit in memory`; the public diff is additive and remains classified
  minor. Integrator=Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`;
  branch=`perf/cuda-packed-lu-split`; lease=none; remaining=independent review,
  PR, merge, and closure; last-update=2026-09-01.

- **Independent review (2026-09-01, Claude):** merged as PR #245;
  `hephaestus_cuda::split_packed_lu` exists (`decomposition/split.rs`) and both
  Python CUDA LU arms (`lu()`, `lu_buffer()`) call it — no host
  `split_packed_lu` round trip remains at the three cited sites. Same 161/161
  debug and release runs as above cover its contracts on hardware.
## HEPH-CHOLESKY-LAZY-HOST-FACTOR [patch] [perf] — done

- **Owner / integrator**: Claude session 5050c72a. Lease:
  `crates/hephaestus-wgpu/src/application/decomposition/cholesky.rs`,
  `crates/hephaestus-wgpu/tests/{contract.rs,contracts.rs}`.
- **Outcome**: retire the eager `n²` host factor materialization that
  `cholesky_decompose_blocked` builds alongside the device factor, so a
  caller that factors and reads `det`/`lower` never pays for it.
- **Driver**: PR #236 moved the closing strict-upper zeroing onto the device
  but left `GpuCholesky::inner`'s host array in place because `det`, `solve`,
  and `inv` read it. Nothing tracked retiring it.
- **Scope**: the WGPU `GpuCholesky` handle and its two entry points.
  **Non-goals**: device-side triangular `solve`, device-side `inv`, the CUDA
  and ROCm sibling handles, and any `CholeskyHandle` signature change.
- **Acceptance oracle**: `det()` stays bitwise equal to
  `leto_ops::CholeskyDecomposition::det` on both entry points (two existing
  contract cases assert this exactly); the captured diagonal is bitwise equal
  to the device factor's diagonal in the multi-panel regime; `solve` after the
  lazy materialization matches the host reference within a derived bound.
- **Risk / change class**: [patch]. No public signature changes, no new
  `HephaestusError` variant (the enum is not `#[non_exhaustive]`, so a variant
  would be [major]); `GpuCholesky`'s fields are private.
- **Dependencies**: none.
- **Verification plan**: warning-denied focused fmt/clippy, WGPU Nextest with
  a real adapter (0 skipped), doctests; a new contract case registered in
  `contracts.rs` with the `CONTRACT_CASES.len()` guard bumped; deliberate
  perturbation of the new device path to prove the case can fail.
- **What forced the host array** (audit, `decomposition/cholesky.rs` at
  `63f5661`): exactly three reads of `GpuCholesky::inner` — `det` :301,
  `solve` :328, `inv` :341-346. The field is private and has no other reader.
  `det` needs only the factor diagonal; `solve` and `inv` need the whole
  `n × n` array for host substitution.
- **Delivered** (`7643ed4`): the blocked path retains the `n`-element factor
  diagonal, captured inside the existing panel loop at zero transfer and zero
  dispatch cost, and `inner` becomes a `OnceLock` the first `solve`/`inv`
  fills by downloading the device factor. The host-delegating entry point
  populates it eagerly as before. Both `n == 0` arms stop building a throwaway
  leto decomposition, and the panel loop's two duplicate scatter/write blocks
  collapse to one.
- **Scoped out, with reason**: device-side triangular `solve` and device-side
  `inv` — each is its own vertical item. A *device reduction* for `det` was
  rejected as the wrong shape: the diagonal is already host-resident inside
  the panel loop, so capturing it costs nothing, whereas a reduction adds a
  dispatch plus a readback per call and reorders the product, breaking the
  bitwise `det` equality two existing contract cases assert.
- **Evidence — transfers/allocations, by inspection**: factorization-time
  device transfers unchanged (the diagonal comes from the per-panel
  `download_matrix_region_compact_into` that already ran). `det`: still 0
  dispatches and 0 readbacks. Host memory no longer materialized per blocked
  factorization: `4n² − 4n = 4n(n−1)` bytes — 143 KiB at n = 192, 255 KiB at
  n = 256 — plus the eliminated scatter of `n²(p+1)/(2p)` host writes over
  `p = ⌈n/64⌉` panels (→ `≈ n²/2`). Accepted cost: the blocked path's first
  `solve`/`inv` adds one `4n²`-byte readback, cached for every later call;
  the host-delegating path adds none.
- **Evidence — differential**: new case
  `blocked_cholesky_retains_factor_diagonal_across_panels` at n = 160 (three
  panels, 64/64/32, ragged tail — regime asserted explicitly, since
  `block_size = BLOCK_SIZE.min(n)` makes any `n ≤ 64` a single SYRK-free
  panel). It asserts `det()` *bitwise* against the product recomputed from the
  downloaded device diagonal, then differentially against leto within
  `4n(n+1)ε ≈ 1.23e-2` relative — derived from Higham (ASNA 2nd ed., Thm 10.3)
  as `4n(n+1)κ∞·u` with `u = ε/2` and `κ∞ ≤ 2` for the strictly diagonally
  dominant fixture — and exercises the deferred materialization through
  `solve` plus a cached repeat solve.
- **Evidence — liveness**: scaling the diagonal capture by `1.000_01` made the
  new case FAIL at `contract.rs:4361` with `left: 2.3870195e23` vs
  `right: 2.3794136e23`, and also tripped the pre-existing
  `blocked_cholesky_identity_yields_identity_lower` (`1.00004` vs `1.0`).
  Note the derived *tolerance* alone did not catch it (3.2e-3 relative sits
  inside the 1.23e-2 bound) — the bitwise assertion is the sharp oracle and
  the tolerance is the backstop. Restored; all cases pass.
- **Evidence — gates**: `cargo fmt -p hephaestus-wgpu -- --check` clean;
  `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings` clean;
  `cargo nextest run -p hephaestus-wgpu` 31 passed, **0 skipped**, 14.553 s
  on a real adapter (`HEPHAESTUS_WGPU_REQUIRE_DEVICE=1`);
  `cargo test -p hephaestus-wgpu --doc` 2 passed;
  `cargo check --workspace --all-targets` clean (metal re-exports this type).
  `CONTRACT_CASES.len()` 173 → 174. Lockfile untouched.
- **Residual**: `solve` and `inv` remain host substitution, so the `n × n`
  download still happens for any caller that uses them. Retiring it needs
  device-side triangular solve — filed below.

- **Closed (stale-claim sweep 2026-09-01, Claude):** delivered as `7643ed4`, merged through PR #242 (2026-09-01); the item still read in-progress. Its remaining `4n²` host download on `solve` is `HEPH-CHOLESKY-DEVICE-TRIANGULAR-SOLVE`.

## ✅ HEPH-CHOLESKY-DEVICE-TRIANGULAR-SOLVE [patch] [perf] — done 2026-09-02

- **Delivered:** `GpuCholesky::solve` runs blocked forward/backward
  substitution on the device (`2·⌈n/256⌉` block solves plus trailing updates
  in one compute pass, RHS copied device-to-device); the `4n²` host-factor
  download and the per-solve RHS round trip are gone from `solve`; `inv` is
  the only remaining host-factor consumer (its `n` device solves are a
  follow-up item). `GpuCholesky::host_factor_materialized` exposes residency.
- **Evidence:** WGPU contract suite 175/175 on RTX 5080 incl. the new
  `n = 300` (256 + ragged 44) case against leto on the shared factor within
  `24γₙ` (Higham Thm 8.5, derivation at the test); mutation proof: zeroing the
  trailing update fails only that case (`x[0]` off by 9e-2 vs bound 4.3e-4)
  while the single-block `2×2` case passes; clippy/doc/doctest/lib green.
- **Residual:** `inv` still downloads the factor — its own item.

## ✅ HEPH-CONFORMANCE-RATCHET-2026-08-31 [patch] — done 2026-08-31

- **Outcome**: restore the Atlas conformance ratchet without raising its
  baseline after provider advances exposed one oversized test sidecar, two
  misclassified test diagnostics, one type-suffixed test helper, one
  wall-clock-synchronized test, and one unbounded release workflow.
- **Scope/non-goals**: WGPU FFT test topology, the WGPU wait-deadline
  regression, and the Python release job bound only; no production numerical
  behavior, wait policy, test workload/assertion, or release trigger changes.
- **Acceptance**: the live provider scan is at or below the committed
  Hephaestus baseline for all five classes; tests use event/poll state rather
  than sleep; moved FFT cases retain their value oracles; the publish job has a
  finite derived bound; warning-denied focused gates pass.
- **Delivered**: PR #240 / merge `94e4695` from source `828de2b` renames the two FFT sidecars into the
  scanner's test namespace, splits the retained/selected-axis cases below the
  500-line target, injects test-only monotonic time for idle decay, and gives
  the two-step PyPI publish job a derived 10-minute bound.
- **Evidence**: exact live scan restores `oversized_files=38`, `print_dbg=0`,
  `type_suffixed_fns=22`, `sleep_synced_tests=0`, and
  `workflow_missing_timeout=0`; standalone exact-lock warning-denied WGPU
  all-target Clippy passes; WGPU Nextest passes 31/31 with no skips in 29.648
  seconds; all workflows parse; rustfmt and diff checks pass.
- **Closure**: hosted host verification, lockfile integrity, and WGPU
  software-adapter contracts pass; independent exact-Git review of source
  `828de2b` / PM head `6d7a0ee` is GREEN, with static-only evidence limits.
  CodeRabbit reports no finding. The external analysis integration errored
  before producing analysis and was not a repository merge gate.
- **Integrator**: Codex `/root/hephaestus_conformance`; lease: none.

## ✅ HEPH-QUALITY-WAVE-2026-08-27 [patch]: Audit-adjudicated safety/perf fixes

- **Delivered**: every accepted finding carries its fix commit on master —
  region-copy error-exit stream drain `31a72a7`, device-to-device
  `clone_cuda_buffer` `7ea5498`, pinned-buffer Deref SAFETY contract `a132613`,
  Python readbacks through `download_owned` `583cfa2`, submission-scoped WGPU
  `copy_buffer` wait `0fe0889`, CUDA context preservation across drop-time
  binds `2476f3c`.
- **Verified 2026-08-31**: each hash confirmed an ancestor of master
  (`git merge-base --is-ancestor`), and the leased regions
  (`decomposition/region.rs`, `infrastructure/pinned.rs`,
  `hephaestus-python/src/backend.rs`) hold those commits as their most recent
  substantive changes. Lease released.
- **Integrator**: closed by Claude session 5050c72a; lease: none.

## HEPH-CUDA-LAUNCH-DRAIN-REEVAL [patch] [perf] — done 2026-09-01

- Owner / integrator: Claude session 03d80d33. Lease: none (discharged).
- **The recorded blocker had expired.** This item was parked on "a Windows host
  with a runtime CUDA device — the current development machine compile-gates
  CUDA only". That is no longer true: the machine now has an RTX 5080 (driver
  610.47, CUDA 13.3, compute 12.0) and `cargo nextest run -p hephaestus-cuda`
  executes 152/152 against it. A blocker is a claim like any other and expires
  by routes other than the one being watched; re-checking it is what unblocked
  this. The device is a GeForce part, so WDDM is the only available driver
  model — the exact configuration the drain targeted.
- **Premise confirmed stale.** The drain cited WDDM's lack of concurrent
  host/device access to `cuMemAllocManaged` ranges. The backend allocates only
  `cuMemAlloc_v2`, and `infrastructure/device.rs` documents that choice as
  deliberately non-managed, so no managed range exists to fault on.
- **Cost, matched pair in one process** (2000 back-to-back launches over 1024
  f32, best of 7 interleaved blocks, single trailing sync so both configs
  measure completed work): **29.750 us/launch with the drain, 6.070 us/launch
  without — 4.9x.** It serialized every launch.
- **Correctness.** A new committed test,
  `cuda_launch_survives_host_allocation_in_flight`, reproduces the exact cited
  scenario — 1500 rounds of host allocate/upload/free with a launch outstanding
  and no intervening synchronization — and does not fault. It asserts output
  values at five indices, so a fault-free run cannot be a run that silently did
  nothing. Package suite 153/153 with `HEPHAESTUS_CUDA_REQUIRE_DEVICE=1` (no
  skips); suite wall time fell from 24.6 s to 21.6 s.
- **Given up deliberately.** The drain's second effect was attributing
  asynchronous kernel-execution faults to the launching operation. Those now
  surface at the next synchronizing operation, exactly as they already did on
  every non-Windows target. Launch *rejection* is unaffected: `cuLaunchKernel`
  reports it synchronously and the return is checked. This trade is recorded at
  the call site, not only here.
- **Unblocks:** KS-7 stream overlap and `HEPH-CUDA-STREAM-ORDERED-ALLOC`, whose
  own dependency line carries the same stale "compile-gates CUDA only" premise
  and should be re-checked before it is treated as blocked.

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

## HEPH-CUDA-STREAM-ORDERED-ALLOC [minor] [perf] — done 2026-09-01

- Owner / integrator: Claude session 03d80d33. Lease: none (discharged).
- **Delivered:** `cuMemAllocAsync`/`cuMemFreeAsync` on the null stream, selected
  per device by a `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` probe recorded on
  `CudaContext`. Allocation and free read the same flag, so they cannot diverge.
  A device without the capability keeps the synchronous pair — a complete
  fallback, not a stub.
- **The blocker had expired here too.** Like the drain item, this carried
  "compile-gates CUDA only". The machine has an RTX 5080; the suite runs against
  it.
- **Cost of the old pair** (`unary_elementwise`, which allocates its output,
  against `unary_elementwise_into`, which reuses a caller buffer — identical
  kernel, so the delta is allocation traffic alone; both blocks end with one
  explicit sync so each measures completed work):

  | elements | allocating | reuse | ratio before | ratio after |
  |---|---|---|---|---|
  | 1024 | 32.1 us | 5.8 us | 5.5x | **1.19x** |
  | 16384 | 34.2 us | 6.0 us | 5.7x | **1.14x** |
  | 262144 | 250.6 us | 7.0 us | 35.8x | **1.01x** |
  | 4194304 | 529.7 us | 14.4 us | 36.8x | **1.07x** |

  Allocation is now approximately free. Reductions, which allocate a fresh
  buffer per pass and retain every one until the reduction ends:

  | elements | passes | before | after | speedup |
  |---|---|---|---|---|
  | 16384 | 2 | 38.8 us | 14.3 us | 2.7x |
  | 262144 | 3 | 42.7 us | 22.1 us | 1.9x |
  | 4194304 | 3 | 304.4 us | 45.3 us | **6.7x** |

- **The win is ordering, not retention.** The pool's release threshold is left
  at its default of zero, so memory still returns to the driver at
  synchronization points and no retained pool grows unbounded. Measured with the
  threshold raised to `u64::MAX` the numbers are the same within noise (4M
  reduction 44.3 us vs 45.0 us), so the retention policy that a hand-rolled pool
  would have required is not needed at all.
- **Drop soundness, the item's stated precondition.** `cuMemFree_v2` was safe
  because it synchronized the whole device. `cuMemFreeAsync` on the null stream
  gives the same guarantee by ordering: the free is enqueued behind work already
  submitted to that stream, and this backend launches every kernel and issues
  every copy on that stream. The ordering is contractual, not incidental, which
  is what this item required before async frees could land. Recorded at the free
  site, and pinned by `cuda_input_buffer_may_drop_while_its_kernel_is_in_flight`
  — 200 rounds dropping a kernel's *input* with the launch outstanding, then
  asserting output values, so an early free shows as corruption rather than
  passing quietly.
- **A claim I had to correct.** The free-site comment first asserted that the
  driver rejects a mismatched alloc/free pair. Forcing the branch showed it does
  not: `cuMemFree_v2` accepts a stream-ordered allocation and the suite passes.
  The pairing is a performance contract, not a validity one, and the comment now
  says so. `cuda_allocation_and_free_select_the_same_allocator` pins the
  structure regardless.
- **Gates:** `hephaestus-cuda` 155/155 with `HEPHAESTUS_CUDA_REQUIRE_DEVICE=1`
  (no skips); `-p hephaestus-core -p hephaestus-host -p hephaestus` 110/110;
  fmt, `clippy --locked --workspace --all-targets -D warnings`, workspace
  doctests, and `cargo doc --locked --workspace --no-deps` all clean.

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

## ✅ HEPH-WGPU-DEFAULT-DEADLINES [major] — done: Bounded default device waits

- **Premise confirmed 2026-08-31** (this fleet's items have carried premises
  that no longer held, so it was re-checked before any edit): `synchronize`
  polled `wgpu::PollType::wait_indefinitely()`, and `download`,
  `download_owned`, `download_sub_buffer`, and `copy_buffer` all reached
  `poll` with `timeout: None`. The audit note understated it by two paths;
  the decomposition region readback (`decomposition/region.rs`
  `wait_for_mappings`) waited indefinitely as well.
- **Delivered**: ADR 0054. `DEFAULT_DEVICE_WAIT` (30 s, derived in-source from
  the Windows TDR envelope — `TdrDelay` 2 s + `TdrDdiDelay` 5 s — so the host
  deadline is the backstop and not the first reporter) now carries all six
  paths; `stage_and_read` and `download_into` take `Duration` rather than
  `Option<Duration>`, so an unbounded wait is no longer expressible there. An
  elapsed deadline surfaces as `HephaestusError::DeviceWaitTimeout { deadline,
  message }`, distinct from `TransferFailed`; no retry, degradation, or
  fallback. No new opt-out surface — `download_with_timeout` and
  `submit_with_timeout` already serve a differing bound, and nothing needed a
  third.
- **Evidence**: the bound is proved to bite by driving the deadline to 1 ns
  behind 512 MiB of queued copy traffic and asserting the *default*
  `download_owned` returns the typed timeout, then recovers. Liveness proved
  twice: ignoring the deadline on the default path fails the case with
  `got Ok([7, 8, 9, 10])`, and mapping `PollError::Timeout` back to
  `TransferFailed` fails it with the wrong variant; it is the only failing case
  either way. The success half is a public-surface contract case (173 cases).
  Gates: fmt, warning-denied all-target Clippy over `hephaestus-wgpu` and
  `hephaestus-core`, workspace `cargo check --all-targets`, nextest 31/31 with
  0 skipped against a real adapter, doctests 2/2.
- **Reclassified [minor] → [major]**: adding a variant to the public
  `HephaestusError` is `enum_variant_added` under `cargo-semver-checks`. Every
  match on that enum in this workspace and in apollo names specific variants
  under a catch-all, so it is source-compatible with all of them; the workspace
  `check --all-targets` is green. See ADR 0054 for why `#[non_exhaustive]` was
  not taken here.
- **Delivered**: PR #239 (`d8dfa26462bb4094568304081f527293edc7034d`).
- **Integrator**: Claude session 5050c72a on `perf/heph-bounded-waits`; lease
  released on merge.

## HEPH-WGPU-SUBMIT-ERROR-SCOPE-WAIT [patch] — done

- Owner: unclaimed.
- Outcome: decide whether `checked_submit`'s error-scope waits can stall
  unboundedly, and bound them if so.
- Evidence (2026-08-31, found while bounding the poll paths): with
  `timeout: None`, `application/prepared.rs` `checked_submit_with_timeout`
  performs no poll at all, so it is not an unbounded *poll* — but it then
  drives three wgpu error-scope futures through `moirai::block_on`
  (`prepared.rs:108-112`), which resolve only once the device processes them.
  Whether that can block indefinitely on a wedged device was not established;
  it is a separate question from the poll deadlines and was deliberately not
  guessed at in ADR 0054.
- Acceptance: the resolution path for error-scope futures traced to its
  wgpu source, and either a bound applied through the same
  `device_wait_deadline()` policy or a recorded argument that the wait is
  already bounded by construction.

- **Resolved (2026-09-01, Claude) — bounded by construction; no code change.**
  Traced to the wgpu 30.0.1 source in the registry,
  `src/backend/wgpu_core.rs` `ContextWgpuCore::pop_error_scope` (~1860-1897):
  on the native `wgpu-core` backend an error scope is a thread-local CPU-side
  stack; `pop_error_scope` pops it synchronously and returns
  `Box::pin(ready(scope.error))` on every path (`ready(None)` on the
  mismatched-thread and unwinding paths). The future is complete before
  `block_on` sees it, so the three `moirai::block_on` calls in
  `prepared.rs:108-112` cannot wait on device progress and cannot stall on a
  wedged device. The only device wait on that path is the explicit
  `PollType::Wait` with the caller's timeout. The web (`webgpu`) backend does
  resolve error scopes asynchronously, but hephaestus does not enable it.
  Re-open trigger: enabling the `webgpu` backend, or a wgpu bump whose
  `wgpu-core` `pop_error_scope` no longer returns a ready future.

## ✅ HEPH-WGPU-STAGING-POOL-DECAY [patch] [perf]: Decay idle staging pool retention

- **Delivered**: PR #229 / implementation `9f39acd` adds age-based idle decay
  to the wgpu staging pool: a shadow retained-byte bound and hit/miss
  counters ride the staging acquire/recycle paths (one clock read plus two
  relaxed loads when warm), and when no staging traffic occurs for
  `STAGING_POOL_IDLE_DECAY` (10 s, derived in-source), the next acquire
  clears parked retention, so a readback burst no longer parks up to 512 MiB
  of MAP_READ memory for the device's lifetime. Uniform pool exempt (~8 KiB,
  reused every dispatch); `clear_transient_pools` resets the bound.
- **Evidence**: a sustained-readback hit-rate test (at most the first of 16
  rounds pays a fresh allocation) and an idle-decay test pinning one fresh
  allocation after the idle window plus pool re-warm on subsequent traffic;
  cargo fmt, warning-denied Clippy, and nextest -p hephaestus-wgpu (25/25)
  pass, and CI Lockfile integrity, Host-side verification, and WGPU feature
  and software-adapter contracts are green.
- **Integrator**: Codebuff session on `perf/wgpu-staging-pool-decay`; lease:
  none.

## HEPH-PY-DEVICE-SIDE-FACTOR-SPLIT [minor] [perf] — done 2026-08-29 (PR #235)

- Integrator: claude-fable session 03d80d33 subagent; last-update: 2026-08-29.
- Lease: crates/hephaestus-wgpu/src/application/decomposition/{split.rs,mod.rs},
  crates/hephaestus-wgpu/src/lib.rs,
  crates/hephaestus-python/src/decomposition.rs, backlog.md.
- Scope: WGPU only. The CUDA path stays on the host round-trip — this host has
  no CUDA device, so a CUDA change would ship unverifiable.
- Outcome: split packed factorization outputs on-device in the Python
  bindings instead of the host round-trip.
- Evidence (audit 2026-08-27): `crates/hephaestus-python/src/decomposition.rs`
  lu/full_piv_lu (and the qr host-factor uploads) download the packed factor,
  split on host (`split_packed_lu`), and re-upload L and U: three full-matrix
  PCIe transfers per factorization. A device-side strided unpack (triangular
  masks writing L and U from the packed buffer) keeps the factors resident.
- Acceptance: no host staging in the factor-split path on either backend,
  value-semantic parity with `split_packed_lu` in differential tests, and
  transfer-count evidence (before/after) attached.
- **Delivered:** PR #235, merged 2026-08-29T21:20Z; member CI green (host
  verification, lockfile integrity, WGPU software-adapter contracts). Closed
  2026-08-31 — the claim had gone stale with the work already landed. The CUDA
  path remains on the host round-trip, as scoped, and is recorded as residual
  rather than assumed equivalent.
- Status: WGPU `lu`/`full_piv_lu` land on `hephaestus_wgpu::split_packed_lu`.
  Instrumented byte counts for the split step go
  3·n²·4 → 0 in both directions, exactly as predicted; wall clock at n=512 is
  ~13× and at n=256 ~2.4–5.4×, while n≤128 stays inside this host's noise band.
  Differential parity against the host oracle is bitwise (copies and structural
  0/1 only, no arithmetic), and the assertion was confirmed to fail under a
  deliberately mutated triangular predicate.
- Residual (acceptance partially met): the CUDA arms still round-trip, so
  "either backend" is unmet by design — see the scope note above.
- **SUPERSEDED — do not file work off the findings below.** They recorded the
  cholesky and qr state as of 2026-08-29; PRs #236, #237, and #238 have since
  retired all three. Verified 2026-08-31 against master:
  - The cholesky `n²` upload is gone — `cholesky.rs` now zeroes the strict
    upper triangle with a device kernel (`hephaestus-cholesky-triangular-zero`,
    PR #236), not `write_buffer(&lower_buf, &host)`.
  - The qr Q claim is retired: `GpuQrDecomposition::accumulate_q` builds Q on
    the device from the stored reflectors (PR #238), and the Python `qr` arm
    returns it instead of uploading `inner().q()`.
  - **The `r_buffer()` contract divergence was proven false** (PR #237). The
    blocked path's panel write-back zeroes every `col < row` entry before the
    buffer reaches the device, so both entry points leave clean upper-triangular
    **R** in `r_buffer()`; only the local name `work_buf` is stale. This is
    asserted, not inferred, by
    `qr_r_buffer_is_upper_triangular_on_both_entry_points`.
- Superseded findings, retained for the audit trail only:
  - `cholesky` in the Python layer never round-tripped; it already returns
    `into_lower()`. The host staging is inside
    `cholesky_decompose_blocked`, whose closing `write_buffer(&lower_buf,
    &host)` exists only to zero the strictly-upper triangle — one n² upload a
    triangular-zero mask would remove. The `host` array itself is not
    removable: `GpuCholesky::inner` needs it for det/solve/inv.
  - `qr` cannot be fixed by the same move. `GpuQrDecomposition` exposes no
    device-resident Q at all, and `r_buffer()` is not substitutable for
    `inner().r()`: `qr_decompose` stores leto's clean R there, but
    `qr_decompose_blocked` stores `work_buf`, the *packed* reflector buffer.
    Removing qr's uploads needs device-side Q accumulation from the stored
    reflectors — a new kernel family, not a contained change. The
    `r_buffer()` contract divergence between the two entry points is a
    latent defect worth its own item.

## ✅ HEPH-WGPU-QR-DEVICE-FACTORS [minor] [perf]: Device-resident R returned; premise corrected

- Integrator: Claude session 5050c72a; lease: none.
- Lease: `crates/hephaestus-wgpu/src/application/decomposition/qr.rs`,
  the `qr` arm of `crates/hephaestus-python/src/decomposition.rs`, the QR
  contract cases, and this item block. The factor-split lease above names
  that Python file, but its last commit is ~6 h old and its WGPU delivery
  merged in #235, so the region is reclaimed per the stale-claim rule; the
  `lu`/`full_piv_lu` arms are untouched.
- **Premise correction (2026-08-30): the `r_buffer()` divergence does not
  exist.** `qr_decompose_blocked` does return the buffer named `work_buf`,
  but the panel write-back zeroes every `col < row` entry before it reaches
  the device (`qr.rs` `factored_panel` and `final_panel` loops), so the
  device buffer holds clean upper-triangular **R**, not packed reflectors.
  The name is stale, the contents are not: the host-side `packed` array fed
  to `QrDecomposition::from_raw_parts` is the separate object that carries
  the reflectors. Both entry points therefore already agree on
  `r_buffer()`'s meaning — asserted directly below rather than inferred from
  `blocked_qr_matches_leto_reference`, whose fixture never reaches the device
  schedule (see the routing trap).
- **Real defect found while checking it:** the Python `qr` binding uploads
  `inner().r()` even though the identical R is already device-resident in
  `r_buffer()` — an `m`x`n` host->device transfer per call that the WGPU
  path had already paid for.
- Outcome (this increment): the redundant R upload is removed, and the
  cross-entry-point `r_buffer()` contract is asserted rather than left
  implicit. Device-side **Q** accumulation is split out below — the half of
  the original premise that survives.
- Acceptance: a contract case exercising both entry points asserts
  `r_buffer()` is upper-triangular and that the two paths agree within a
  derived tolerance; the Python `qr` R transfer count drops by `4mn` bytes
  with the returned R value-identical to the uploaded one.
- **Routing trap found in review (2026-08-30):** `qr_decompose_blocked`
  delegates to `qr_decompose` at or below `QR_DIRECT_PANEL_LIMIT` (4) panels
  of `QR_BLOCK_SIZE` (32) columns, so every QR fixture at `n = 35` — including
  `blocked_qr_matches_leto_reference`, whose comment claims it "exercises two
  QR blocks" — runs the *host* path. The first version of the case below
  inherited that shape and so compared the delegating path against itself:
  its assertions held vacuously. The case now covers both regimes, `n = 35`
  (2 panels, delegating) and `n = 129` (5 panels, device schedule), with a
  guard asserting each fixture lands where intended.
- **Delivered 2026-08-30:** `qr_r_buffer_is_upper_triangular_on_both_entry_points`
  asserts, at both shapes, that both paths' `r_buffer()` is zero below the
  diagonal and that they agree within `2·c(m,n)·ε·5.1` from Householder
  columnwise backward stability. It also pins the identity the Python change
  rests on — the device buffer equals `inner().r()` **exactly**, since each R
  row is produced once and stored to both the host `packed` array and the
  buffer — measured bitwise on the device route, not merely within tolerance.
  Liveness confirmed: with the panel write-back's `col < row` zeroing
  disabled, the case fails at `R[129, 128] = 4.99e-3` (a reflector surviving
  in the tail region) and `blocked_qr_preserves_panel_boundary_contracts`
  fails with it; the `n = 35` arm alone catches neither. `GpuQrDecomposition::into_r_buffer` hands that
  buffer over, and the Python `qr` arm returns it instead of re-uploading
  `inner().r()`: **`4mn` bytes removed per WGPU `qr` call** (9.8 KiB at the
  test shape, 4 MiB at m=n=1024), with the Q upload left in place and its
  removal split into `HEPH-WGPU-QR-DEVICE-Q`. No wall-clock claim.
- **Gates:** `cargo nextest run -p hephaestus-wgpu -p hephaestus-core`
  139/139, 0 skipped, against a real adapter (171 contract cases);
  warning-denied all-target Clippy over `hephaestus-wgpu` and
  `hephaestus-python`; doctests and `cargo fmt --check` clean.

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

## ✅ HEPH-WGPU-CHOLESKY-TRIANGLE-MASK [patch] [perf]: Device-side triangular zero

- **Delivered**: `cholesky_decompose_blocked`'s closing
  `write_buffer(&lower_buf, &host)` is replaced by `zero_strict_upper`, a
  256-wide WGSL pass that zeroes only the cells with `col > row` in place.
  `GpuCholesky::inner` keeps its host array unchanged — det/solve/inv still
  read it — so only the redundant upload goes.
- **Byte-count evidence**: the removed call uploaded the whole `n`x`n` f32
  factor, `4n^2` bytes per factorization — 256 KiB at n=256, 1 MiB at n=512,
  4 MiB at n=1024 — of which every cell at or below the diagonal was
  byte-identical to what the device already held. Host→device traffic in the
  blocked path now ends with the per-panel scatters. No wall-clock claim is
  made: the transfer count is exact by inspection, and this host's timing
  noise was not separated from it.
- **Correctness evidence**: new contract case
  `blocked_cholesky_zeroes_strict_upper_outside_diagonal_blocks` (n=96 > the
  64-element block, off-diagonal fixture value 0.5) asserts exact zeros above
  the diagonal plus a positive diagonal, so an all-zero buffer cannot pass.
  Liveness proven: with the mask predicate deliberately disabled, it fails at
  `[0, 64]` holding the input's `0.5` — the first cell outside the first
  diagonal block — and the pre-existing
  `blocked_cholesky_matches_leto_reference_across_block_boundary` and
  `..._spd_reconstruction_matches_original` fail alongside it. Restored, all
  170 contract cases pass.
- **Gates**: `cargo nextest run -p hephaestus-wgpu` 31/31 with 0 skipped
  against a real adapter, doctests 2/2, warning-denied all-target Clippy, and
  `cargo fmt --check` clean.
- **Integrator**: Claude session 5050c72a; lease: none.
- Equivalence confirmed before implementing: `panel_cholesky_packed`
  (`hephaestus-core/src/domain/decomposition.rs:238-241`) already zeroes each
  diagonal block's strictly-upper triangle, and the per-panel uploads cover
  every cell with `row >= blockstart(col)`. The closing `write_buffer` is
  therefore bitwise identical to the device state except on the strict upper
  triangle outside the diagonal blocks, which still holds the entry copy's
  input values. A strict-upper zero pass is an exact replacement, not an
  approximation.

- Outcome: `cholesky_decompose_blocked` zeroes its strictly-upper triangle on
  the device instead of uploading the assembled host matrix.
- Evidence (2026-08-29):
  `crates/hephaestus-wgpu/src/application/decomposition/cholesky.rs` closes with
  `device.write_buffer(&lower_buf, &host)` — an n² host→device upload whose only
  effect beyond the per-panel scatters already performed is zeroing the strict
  upper triangle, which still holds the input's values from the entry copy.
- Acceptance: the upload is replaced by a device-side triangular-zero pass, the
  existing Cholesky contracts still pass against a real adapter, and byte-count
  evidence shows the n² host→device transfer removed. `GpuCholesky::inner`
  keeps its host array; only the redundant upload goes.

## HEPH-CUDA-OXIDE-MEMCPY2D-ABI [patch] — todo (external blocker)

- Owner: unclaimed; root cause is upstream (cuda-oxide), outside the
  allowlist — external integration requirement.
- Outcome: replace the per-row 2-D region-copy workaround with one
  `cuMemcpy2DAsync` per region once upstream fixes the ABI.
- Evidence (audit 2026-08-27):
  `crates/hephaestus-cuda/src/application/decomposition/region.rs` enqueues
  per-row 1-D copies because cuda-oxide 0.4.0 generates `size_t` as
  `c_ulong`, making `CUDA_MEMCPY2D` layout-incompatible with the CUDA driver
  ABI on Windows/MSVC (module doc records this).
- Re-open trigger: a cuda-oxide release with a corrected `CUDA_MEMCPY2D`
  layout on Windows/MSVC; verify the struct layout against the driver header
  before adopting.

## ✅ HEPH-CUDA-LIMITS-SEMANTICS [minor] — done 2026-09-02

- **Delivered:** `query_device_limits` reports total device memory as `max_buffer_size` (the stable per-device capacity, matching the WGPU hard limit's semantics) and `CudaDevice::free_memory_bytes` is the point-in-time runtime query. `require_limits` therefore compares against a value that no longer decays with allocations.
- **Consumer sweep:** `max_buffer_size` readers are hephaestus-internal (`require_limits`, ROCm/Metal/host contract tests) plus coeus-wgpu's output-size validation, which reads the WGPU limit, not CUDA's — no caller read the CUDA value as "free now", so no migration.
- **Evidence:** RTX 5080 — `device_capabilities_are_driver_backed` asserts free ≤ capacity; the new `buffer_limit_is_stable_across_allocations_and_free_memory_tracks_them` allocates 64 MiB and sees the limit unchanged and free memory fall; hephaestus-cuda contract 87/87, clippy and rustdoc clean.
- **Follow-up 2026-09-02 (concurrent duplicate reconciled).** A second agent
  implemented the same correction independently; its delivery is superseded
  by the one above except for one clause, ported here. Neither value clause
  discriminates the defect: the old field was a free reading *stored at
  acquisition*, so it was stale rather than moving, and
  `buffer_limit_is_stable_across_allocations_and_free_memory_tracks_them`
  **passes on the defective code** (verified by reintroducing it).
  `the_buffer_limit_is_built_from_total_device_memory_not_the_free_reading`
  asserts on the source which half of `cuMemGetInfo_v2` reaches
  `DeviceLimits`, fails on that reintroduction, and needs no device, so it
  guards on every runner rather than only where a GPU is present.

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

## ✅ HEPH-WGPU-TEST-DEVICE-REUSE-1 [patch] [perf]: Reuse WGPU test devices

- **Delivered**: PR #228 / implementation `5507479` consolidates 242 semantic cases from 26 binaries and 242 process tests into two binaries and 25 process tests without changing case workloads, assertions, buffer isolation, optional-feature behavior, or intentional cross-device acquisition.
- **Evidence**: exact package execution falls from 76.936 to 27.168 seconds (64.7%) with no skips; host Clippy, Windows AArch64 all-target checking, no-default library checking, Rustdoc, and 2/2 doctests pass, and two independent static reviews are GREEN. The pre-existing no-default all-target failures remain outside this increment.
- **Integrator**: Codex session `01a0253c-6013-7552-99cc-36bbbcf77f6d`; lease: none.

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

## HEPH-CUDA-F64-COMPARISON-1 [minor] — done 2026-08-13

- Owner: Atlas integration; scope: the provider-owned typed CUDA comparison
  expressions and CUDA contract instantiation. The Coeus consumer refresh is a
  separate dependent increment after this provider merge.
- Acceptance: `Eq`, `Ne`, `Lt`, `Gt`, `Le`, and `Ge` compile for `CudaC, f64`
  with double-valued result masks; no host or legacy NVRTC path is added.
- Verification: the merged provider head `b34b50787df636891d281b5011c6a17dd46edcb0`
  contains the shared six-operation `CudaC, f64` typed expression contract and
  the CUDA value-semantic instantiation test. Exact-head hosted runs
  `31669318548` (CUDA), `31669318571` (WGPU), `31669318539` (ROCm), and
  `31669318537` (Metal) all pass. These runs establish hosted backend CI only;
  no physical CUDA-device execution claim is added. PR #204 merged the
  implementation at `b34b507`; the remaining Coeus consumer cutover is tracked
  separately.

## HEPH-CROSS-ENTROPY-PROVIDER-1 [minor] [arch] — done 2026-08-12

- Owner: Codex. The implementation merged through PR #192 at `1e1f12c`;
  this closeout records the final exact-head evidence. Scope: device-neutral
  cross-entropy planning, WGPU/CUDA/ROCm/Metal implementations, conformance,
  and owner-keyed PM/ADR records.
- Outcome: execute stable mean cross-entropy forward and additive backward on
  the selected accelerator while probabilities and gradients remain provider
  resident.
- Non-goals: CPU execution, consumer autograd wiring, silent provider changes,
  class weighting or ignored labels, and performance claims without matched
  measurement.
- Dependencies: merged Leto 0.40 cross-entropy oracle and its Eunomia 0.8/rkyv
  0.8 provider graph; WGPU 30, cuda-oxide 0.4, and the existing HIP toolchain.
- Acceptance: one core loss seam validates complete requests before dispatch;
  all four providers pass shared f32 forward/backward and typed rejection
  contracts without host payload transfer or fallback; warning-denied gates,
  Nextest, doctests, SemVer checks, independent review, and exact-head provider
  CI pass before merge.
- Status: provider seam, implementations, conformance, warning-denied checks,
  doctests, focused Nextest, core SemVer, independent review, and exact-head
  hosted provider CI are complete. The final WGPU, CUDA, ROCm, Metal, and
  mdBook workflows pass at `bc6dfcf` (`31646386129`, `31646386243`,
  `31646386123`, `31646386192`, `31646386586`). The local full
  affected-package launch did not reach test execution within its five-minute
  shared-cache contention budget and is not counted as evidence.

## HEPH-WGPU-PREPARED-WORK-1 [patch] [perf] — done

- Owner: Codex on `codex/perf-wgpu-prepared-work`; scope: WGPU prepared scalar
  reduction work-state representation, empty-dispatch benchmark and contracts,
  and owner-keyed PM/release records.
- Outcome: encode empty, singleton-copy, and reduction-tree work as one valid
  state; hoist type-independent command encoding out of `<T>`; and make direct
  empty prepared dispatch return before command-encoder allocation or queue
  submission.
- Non-goals: reduction arithmetic or geometry; CUDA/ROCm/Metal kernels; public
  API changes; KS-5 decomposition files; and runtime or memory claims without
  matched measurement or structural evidence.
- Acceptance: empty, singleton, and multi-pass values remain exact; the direct
  empty-dispatch benchmark records matched samples; source structure cannot
  represent conflicting prepared work; focused Nextest, warning-denied Clippy,
  formatting, doctests, independent review, and exact-head WGPU/CUDA/ROCm/macOS
  Metal CI pass.
- Risk/change class: `[patch] [perf]`; internal WGPU state representation and
  no-work dispatch routing only. Stop condition: reject the representation if
  it increases `PreparedReduction<u32>` size or regresses non-empty contracts.
- Status: implementation and focused local verification complete. Empty,
  singleton, multi-pass, and mixed value contracts pass 2/2; warning-denied
  default all-target and no-default library Clippy pass. Three matched samples
  reduce the direct-empty median from 19.773 microseconds to 1.032 nanoseconds;
  the measured inline host plan size falls from 80 to 72 bytes. Independent
  review initially rejected warm-up accounting, iteration count, and ambiguous
  size labeling; the calibrated harness fixes all three, and re-review reports
  no remaining finding. No-default doctests pass 2/2. Exact implementation-head
  provider CI passes at `21105e3`: CUDA `30783614385`, ROCm `30783614348`, WGPU
  `30783614399`, and macOS Metal `30783614354`; hardware-only AMD and NVIDIA
  jobs skip as designed on unlabelled hosted runners.

## HEPH-WGPU-AXIS-TILE-2 [patch] [perf] — done

- Owner: Codex on `codex/perf-wgpu-axis-tile`; scope: WGPU rank-2 axis-0
  reduction tile geometry, focused value contracts, matched comparative
  measurements, and owner-keyed PM/release records.
- Outcome: reduce fixed GPU dispatch and workgroup-barrier overhead for the
  measured 256x256 axis-0 workload without adding buffers, submissions, CPU
  routing, or backend-specific public API.
- Non-goals: CPU fallback; fused multi-statistic API; arithmetic changes;
  CUDA/ROCm/Metal kernels; KS-5 decomposition files; and performance claims
  without matched measurements.
- Acceptance: exact axis sum/min/max/mean values remain unchanged; a source
  contract pins valid tile geometry; three matched benchmark samples show a
  stable improvement or the experiment is rejected; formatting,
  warning-denied Clippy, focused Nextest, independent review, and exact-head
  WGPU/CUDA/ROCm/macOS Metal CI pass.
- Risk/change class: `[patch] [perf]`; internal WGPU dispatch geometry only.
  Stop condition: retain the current 32-column tile if a wider geometry does
  not improve the unchanged workload without semantic regression.
- Status: the 64-column geometry, host shape contract, and physical WGPU
  256x256 exact sum/min/max/mean plus narrow-width regression pass locally.
  Three interleaved samples reduce the single-reduction median from 55.522 to
  39.212 microseconds and the eight-reduction median from 28.494 to 25.286
  microseconds. Warning-denied all-target Clippy, focused Nextest, doctests,
  formatting, and the comparative benchmark pass through the local Atlas
  overlay. Independent re-review is clean after the physical multi-tile
  regression closed its initial coverage finding. Exact implementation-head
  WGPU run `30777848642`, CUDA run `30777848679`, ROCm run `30777848659`, and
  native macOS Metal run `30777848666` pass at `eeebcf5`; PR #190's
  documentation-only closeout head remains the merge gate.

## HEPH-CUDA-COPY-SYNC-1 [patch] [perf] — done

- Owner: Codex on `codex/perf-cuda-copy-sync`; scope: CUDA's synchronous
  device-local `ComputeDevice::copy_buffer` path, focused transfer contracts,
  and owner-keyed PM/release records.
- Outcome: preserve host-visible copy completion while replacing context-wide
  synchronization after `cuMemcpyDtoD_v2` with a default-stream wait.
- Non-goals: asynchronous copy semantics; stream API redesign; WGPU/ROCm/Metal;
  transfer volume; KS-5 decomposition files; and runtime claims without matched
  CUDA hardware measurements.
- Acceptance: `copy_buffer` returns after the CUDA device-to-device copy without
  calling `cuCtxSynchronize`; a default-stream wait preserves host completion;
  exact values, zero-length copies, and typed length rejection remain unchanged;
  an adapterless source contract prevents the global barrier or an omitted wait
  from returning; focused Nextest, warning-denied Clippy,
  formatting, independent review, and exact-head WGPU/CUDA/ROCm/macOS Metal CI
  pass.
- Risk/change class: `[patch] [perf]`; internal synchronization plus zero-byte
  transfer correctness. Nonzero CUDA copy calls and bytes, buffer ownership,
  and the public API remain unchanged.
- Status: implementation and focused local verification complete 2026-08-02.
  `copy_buffer` retains one device-local copy and stream submission, then waits
  on the default stream instead of the whole context. Physical CUDA exact-value, empty-copy,
  typed mismatch, and shared transfer conformance pass locally. The broader
  transfer gate exposed and closed zero-sized POD allocation/transfer calls
  while preserving their logical lengths. The adapterless regression pins the
  driver call, rejects its async counterpart, and requires stream-scoped
  completion. Final physical transfer conformance and formatting pass.
  Independent re-review approves with no remaining findings. Exact-final-diff
  Clippy/doctest collection timed out behind peer-held shared-target locks.
  Exact implementation-head WGPU run `30774973252`, CUDA run `30774973245`,
  ROCm run `30774973255`, and native macOS Metal run `30774973240` pass and
  supply the warning/doc oracle. Hardware-only NVIDIA and AMD jobs skip because
  matching labeled runners are unavailable; PR #188's documentation-only
  closeout head remains the merge gate.

## HEPH-ROCM-COPY-SYNC-1 [patch] [perf] — done

- Owner: Codex on `codex/perf-rocm-copy-sync`; scope: ROCm's synchronous
  device-local `ComputeDevice::copy_buffer` path, focused transfer contracts,
  and owner-keyed PM/release records.
- Outcome: preserve the synchronous copy contract while removing the redundant
  whole-device synchronization performed after synchronous `hipMemcpyDtoD`.
- Non-goals: asynchronous copy semantics; stream API redesign; WGPU/CUDA/Metal;
  transfer volume; KS-5 decomposition files; and runtime claims without matched
  ROCm hardware measurements.
- Acceptance: `copy_buffer` returns after the HIP device-to-device copy without
  calling `hipDeviceSynchronize`; exact values, zero-length copies, and typed
  length rejection remain unchanged; an adapterless source contract prevents
  the global barrier from returning; focused Nextest, warning-denied Clippy,
  formatting, independent review, and exact-head WGPU/CUDA/ROCm/macOS Metal CI
  pass.
- Risk/change class: `[patch] [perf]`; internal synchronization only. The HIP
  copy call, transfer bytes, buffer ownership, and public API remain unchanged.
- Status: implementation and focused local verification complete 2026-08-02.
  `copy_buffer` retains one synchronous device-local copy and stream submission
  without a following global barrier. The corrected adapterless regression
  pins the synchronous HIP call and rejects its async counterpart. The complete
  no-default-feature contract and transfer binaries pass 37/37, all-target
  warning-denied Clippy and doctests pass, and formatting is clean. Physical
  value execution remains hosted ROCm evidence. Independent re-review approves
  with no remaining findings. Exact implementation-head WGPU run `30772689478`,
  CUDA run `30772689476`, ROCm run `30772689469`, and native macOS Metal run
  `30772689455` pass. Hardware-only NVIDIA and AMD jobs skip because matching
  labeled runners are unavailable; PR #187's documentation-only closeout head
  remains the merge gate.

## HEPH-ROCM-SPARSE-READBACK-1 [patch] [perf] — done

- Owner: Codex on `codex/perf-rocm-sparse-readback`; scope: ROCm CSR
  device-to-host reconstruction, its focused source/value contracts, and
  owner-keyed PM/release records.
- Outcome: route CSR values, column indices, and row pointers through the
  provider-owned readback seam, removing three initialized host-vector writes
  that successful HIP transfers wholly overwrite.
- Non-goals: sparse arithmetic or storage layout; transfer count or volume;
  WGPU/Metal/CUDA; KS-5 decomposition files; and runtime or peak-memory claims
  without matched hardware measurements.
- Acceptance: all three full-vector CSR readbacks use `download_owned`; empty
  and non-empty round trips retain exact Leto values; a source regression
  prevents initialized readbacks from returning; focused Nextest,
  warning-denied Clippy, formatting, independent review, and exact-head
  WGPU/CUDA/ROCm/macOS Metal CI pass.
- Risk/change class: `[patch] [perf]`; host initialization only. Allocation
  count, transfer volume, device storage, and public API remain unchanged.
- Status: implementation and focused local verification complete 2026-08-02.
  All three CSR vectors use provider-owned readback. The adapterless source
  contract passes 1/1, and the focused sparse contract remains green 1/1;
  physical value execution is runner-gated on this Windows host. Formatting,
  no-default-feature all-target warning-denied Clippy, doctests, and Rustdoc
  complete; Rustdoc retains two pre-existing unrelated broken links.
  Independent review approves with no findings and notes the documented lack
  of local physical ROCm execution. Exact implementation head `922b05c` passes
  WGPU run `30770979399`, CUDA run `30770979420`, ROCm run `30770979416`, and
  native macOS Metal run `30770979385`. Hardware-only NVIDIA and AMD jobs skip
  because this dispatch did not request self-hosted devices. PR #186's
  docs-only closure head repeats the provider matrix as the merge gate.

## HEPH-METAL-ACQUISITION-1 [minor] — done

- Owner: Codex on `codex/feat-metal-device-acquisition`; scope: Metal's shared
  device-acquisition trait implementation, its WGPU Metal-only substrate,
  focused conformance, and owner-keyed PM/release records.
- Outcome: make Metal substitutable through `ComputeDeviceAcquisition`, honoring
  device preference, optional features, required limits, and bounded
  enumeration without selecting a non-Metal adapter.
- Non-goals: KS-5 decomposition files; kernel behavior; CUDA/ROCm acquisition;
  backend fallback; and runtime or memory claims.
- Acceptance: `MetalDevice` implements the shared seam; every acquired device
  is Metal-backed; optional features are intersected with adapter support;
  required limits are enforced; zero-device enumeration returns no device;
  focused Nextest, warning-denied Clippy, doctest/Rustdoc, independent review,
  and exact-head WGPU/CUDA/ROCm/macOS Metal CI pass.
- Risk/change class: `[minor]`; additive backend capability and WGPU substrate
  API, with no existing constructor behavior change.
- Status: implementation and focused exact-source verification complete
  2026-08-02. Metal implements the shared seam through Metal-only WGPU adapter
  selection; WGPU and Metal enumeration now honor preference ordering and
  return immediately for a zero-device bound. Focused acquisition contracts
  pass 3/3 and the WGPU preference-policy unit passes 1/1; both affected crates
  compile and pass all-target warning-denied Clippy, doctests, warning-clean
  Rustdoc, and formatting. WGPU SemVer passes 196/196 applicable checks; Metal
  SemVer is blocked before API analysis by an unexpected MSVC linker failure in
  the tool's temporary rustdoc graph. Independent review approves after
  requiring one-time limit selection, viable-adapter continuation, physical
  identity/feature assertions, and preservation of `try_default`; all are
  applied. Exact implementation head `3b8cc85` passes WGPU run `30762519498`,
  CUDA run `30762519504`, ROCm run `30762519499`, and native macOS Metal run
  `30762519503`. Hardware-only NVIDIA and AMD jobs skip because this dispatch
  did not request self-hosted devices.

## HEPH-WGPU-DECOMP-READBACK-1 [patch] [perf] — done

- Owner: Codex on `codex/perf-wgpu-decomposition-readback`; scope: WGPU
  non-blocked decomposition full-vector readbacks, their source/value contract,
  Metal inheritance, and owner-keyed PM/release records.
- Outcome: route host-delegated decomposition inputs and solve right-hand sides
  through `ComputeDevice::download_owned`, removing the initialized host-vector
  pass that mapped staging wholly overwrites.
- Non-goals: `lu`, `qr`, and `cholesky` files claimed by KS-5; core host-loop
  consolidation; CUDA/ROCm; decomposition arithmetic; transfer staging policy;
  and runtime or peak-memory claims without matched measurements.
- Acceptance: every direct full-vector readback in the nine claimed families is
  provider-owned; empty, invalid, factorization, solve, and reconstruction
  values remain unchanged; a source regression prevents initialized heap
  readbacks from returning; focused Nextest, warning-denied Clippy, formatting,
  independent review, and exact-head WGPU/macOS Metal CI pass.
- Risk/change class: `[patch] [perf]`; host allocation initialization only.
  Allocation count, transfer volume, and device storage remain unchanged.
- Status: implementation and focused exact-source verification complete
  2026-08-02. Sixteen matrix-input and solve-vector readbacks now use WGPU's
  failure-atomic `download_owned`; the syntax-aware source regression and shared
  decomposition value contract pass 2/2 under the committed Git graph. The
  Atlas-overlay attempt failed before Hephaestus compilation because fresh
  peer-owned Moirai `main` enabled `missing_docs` while its async feature
  surface remained incomplete; the locked CI-equivalent graph at Moirai
  `b7988419` compiles and tests cleanly. Formatting, all-target warning-denied
  Clippy, doctests 2/2, and warning-clean Rustdoc pass. Independent review
  approves the corrected syntax-aware regression. Exact implementation head
  `d27cfd6` passes WGPU run `30760402397`, CUDA run `30760402388`, ROCm run
  `30760402390`, and native macOS Metal run `30760402389`. Hardware-only NVIDIA
  and AMD jobs skip because this dispatch did not request self-hosted devices.

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

## HEPH-WGPU-METAL-OWNED-READBACK-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-wgpu-owned-readback`; scope: WGPU's
  synchronous staging readback, Metal delegation, shared transfer contracts,
  and owner-keyed PM/release records.
- Outcome: let WGPU and Metal initialize reserved host result capacity directly
  after successful mapped staging, removing the initialized zero-fill pass from
  `download_owned` while preserving the existing `download` contract.
- Non-goals: staging-pool policy, asynchronous transfers, CUDA/ROCm transfer
  implementations, and active Python/KS-7 scopes.
- Acceptance: one staging/map orchestration remains authoritative; owned
  transfer publishes vector length only after every byte is copied; empty and
  zero-sized values are safe; bit patterns survive round-trip; focused
  warning-denied gates, independent review, and exact-head WGPU/macOS Metal CI
  pass.
- Delivery-blocker update: Themis changed its Cargo package identity to
  `themis-topology` at upstream `a1c8231` while retaining the Rust library name
  `themis`. Fresh hosted resolution therefore failed before provider builds.
  This item absorbs the required dependency-alias cutover and lock/overlay
  refresh; no backend API changes result.
- Status: done 2026-08-02. Exact implementation head `1c4ac16` passes WGPU run
  `30735730194`, CUDA run `30735731839`, ROCm run `30735732609`, and native
  macOS Metal run `30735730960`. Hardware-only CUDA and ROCm jobs skipped
  because this dispatch did not request self-hosted devices.

## HEPH-DECOMP-OWNED-READBACK-1 [patch] [perf] — done

- Owner: Codex on shared `feat/topology-option`; scope: CUDA and ROCm
  decomposition modules that allocate an initialized host vector immediately
  before a successful full-buffer device download, plus focused contracts and
  release records. Active topology/device files and WGPU/Metal are excluded.
- Outcome: route decomposition readbacks through the provider-owned
  `ComputeDevice::download_owned` implementation so CUDA/HIP copies publish
  initialized vectors only after success and do not first write zero across
  host storage that the device transfer wholly overwrites.
- Acceptance: every in-scope direct full-buffer readback uses `download_owned`;
  empty, invalid, factorization, solve, and decomposition values remain
  unchanged; source regression prevents reintroduction; focused Nextest,
  feature checks, warning-denied Clippy, and exact-head backend CI pass.
- Risk/change class: `[patch] [perf]`; internal allocation placement only.
  Allocation count and transfer volume remain unchanged. Runtime and peak-memory
  changes require matched measurements and are not claimed.
- Status: implementation and focused local verification complete 2026-08-01.
  Twenty-five CUDA and 28 ROCm heap-vector readbacks now use the provider-owned
  result; seven ROCm one-word status/rank stack reads retain `download`.
  Physical CUDA decomposition conformance plus the structural regression pass
  2/2; adapterless ROCm structural conformance passes 1/1. Warning-denied
  decomposition Clippy and full-tree formatting pass. Independent review
  initially rejected a substring-based stack-read exception; exact-call and
  exact-array guards were added, focused tests rerun, and re-review approves.
  Implementation head `b685cc6` passes CUDA run `30731767939`, ROCm run
  `30731767934`, WGPU run `30731767943`, and native macOS Metal run
  `30731767923`; hardware-only NVIDIA and AMD jobs skip because no runner was
  dispatched. The external `recurseml/analysis` service reports its generic
  error and is not provider evidence. Final docs-only exact-head CI remains PR
  #178's merge gate. Rustdoc builds with four pre-existing unrelated broken-link
  warnings across the CUDA/ROCm stencil and sparse modules.

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
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## HEPH-OWNED-DOWNLOAD-1 [minor] [perf] — done

- Owner: Codex on `codex/hephaestus-owned-download`; scope: the
  `ComputeDevice` owned-download seam, shared transfer conformance, optimized
  CUDA/ROCm implementations, and CUDA/ROCm `pinv`/`matexp` consumers.
- Outcome: let a provider allocate device-to-host results so CUDA and ROCm do
  not zero-fill host vectors immediately before a synchronous full overwrite.
- Non-goals: transfer arithmetic, device staging-pool policy, WGPU/Metal
  zero-fill removal, native GPU implementations of host-delegated matrix
  functions, active attention/parameterized-unary/stateful scopes, or runtime
  and peak-memory claims without controlled measurements.
- Acceptance: `download_owned` is default-compatible for external backends,
  bitwise-preserving and empty-safe across shared conformance; CUDA/ROCm publish
  no partially initialized vector on failure and their four matrix-function
  downloads use the provider-owned result directly; focused value contracts,
  feature-off stubs, warning-denied gates, and exact-head backend CI pass.
- Risk/change class: `[minor]` additive open-trait API and `[perf]` removal of
  four redundant host initialization writes. Unsafe length publication is
  confined to synchronous CUDA/ROCm transfers after every byte is written.
- Status: implementation complete 2026-08-01. Pre-edit WGPU matrix-function
  baseline passes 8/8 and physical CUDA passes 2/2. Post-edit shared transfer
  conformance passes on WGPU and physical CUDA; CUDA matrix-function contracts
  pass 3/3, including empty outputs; adapterless ROCm differential, closed-form,
  invalid, and empty contracts pass 3/3. Feature-enabled and feature-off
  warning-denied checks, formatting, doctests, and Rustdoc build pass.
  Independent review approves the failure-atomic copy design after requiring a
  safe ZST path plus changelog entry; both are applied and re-review approves.
  `cargo-semver-checks` cannot retrieve a registry baseline because
  `hephaestus-core` is unpublished. Implementation head `dad12d6` passes CUDA
  run `30713948491`, ROCm run `30713948448`, WGPU run `30713948446`, and macOS
  Metal run `30713948456`; hardware-only NVIDIA and AMD jobs skip because no
  runner was dispatched. The external `recurseml/analysis` service reports its
  generic error and is not provider evidence. Final docs-only exact-head CI
  remains PR #175's merge gate.

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

## HEPH-PREPARED-L2-OVERWRITE-1 [patch] — done

- Owner: Codex on `codex/hephaestus-prepared-l2-overwrite`; scope: CUDA and
  ROCm prepared L2 result allocation, existing prepared-map value contracts,
  and synchronized PM evidence.
- Outcome: preserve defined CUDA/ROCm prepared-L2 output across the public
  pre-dispatch and failed-dispatch lifecycle, strengthen successful-dispatch
  overwrite coverage, and correct the stale overwrite-performance claim.
- Non-goals: map/reduction arithmetic, empty identities, plan capacities,
  WGPU/Metal platform allocation, active attention/parameterized-unary/stateful
  update scopes, or runtime claims without matched hardware measurements.
- Acceptance: both public strided and internal dense CUDA/ROCm preparation
  paths retain zeroed scalar storage while the unary kernel demonstrably
  assigns the sole element on every successful dispatch; repeated-input,
  strided, empty, and exact-value contracts remain green; warning-denied
  provider gates and exact-head CUDA, ROCm, WGPU, and macOS Metal CI pass.
- Risk/change class: `[patch]` correctness and evidence repair. The rejected
  uninitialized allocation would expose undefined contents through `output()`
  before first dispatch or after failure. Allocation count, peak memory,
  arithmetic, and public ownership remain unchanged; no performance gain is
  claimed.
- Status: implementation and focused local gates complete 2026-08-01.
  Physical CUDA baseline and post-edit contracts pass 1/1, including `NaN`
  poisoning before first and repeated dispatch; adapterless ROCm compiles and
  passes its typed-unavailable contract. Formatting, feature-enabled CUDA and
  adapterless ROCm warning-denied Clippy, and no-default all-target checks pass.
  Independent review rejected the proposed uninitialized allocation because
  `output()` is public before first dispatch and remains accessible after a
  failed dispatch. The implementation therefore retains all four defined
  allocations, keeps the stronger successful-overwrite contracts, and corrects
  the changelog boundary. Independent re-review approves the corrected
  lifecycle, tests, and evidence claims. Implementation head `998f521` passes
  CUDA run `30710893281`, ROCm run `30710893299`, WGPU run `30710893294`, and
  macOS Metal run `30710893263`; hardware-only NVIDIA and AMD jobs skip because
  no runner was dispatched. The external `recurseml/analysis` integration
  reports its generic service error and is not provider evidence. Final
  docs-only closeout-head CI remains PR #173's merge gate.

## HEPH-MATPOW-DEVICE-IDENTITY-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-matpow-device-identity`; scope: WGPU,
  CUDA, and ROCm matrix-power identity initialization, shared value contracts,
  Metal inheritance through WGPU, and synchronized PM evidence.
- Outcome: construct the exponent-zero identity directly in device storage so
  matrix power allocates no matrix-sized host vector and performs no
  matrix-sized host-to-device identity upload.
- Non-goals: exponentiation order, matrix-multiply kernels, base/scratch
  allocation policy, scalar precision, layout semantics, or unmeasured runtime
  claims.
- Acceptance: square-shape and checked-size validation precede allocation; one
  backend-native dispatch assigns every identity element; empty and
  exponent-zero results retain exact values; no host matrix-sized identity
  allocation remains; exact-head CUDA, ROCm, WGPU, and macOS Metal CI passes.
- Risk/change class: `[patch]` internal initialization placement. Peak device
  allocation is unchanged; host peak memory and identity transfer volume fall
  from `O(n^2)` to `O(1)` by construction.
- Evidence: WGPU, CUDA, and ROCm compile warning-clean with distinct cached,
  scalar-generic identity kernels. Local WGPU device execution passes 2/2
  focused matrix-power contracts, including trait-defined nonstandard identity
  values, exact built-in exponent-zero identity, empty, strided, odd-power,
  and non-square behavior. Adapterless CUDA/ROCm focused runs pass 3/3 compile
  and source contracts; their value-semantic execution remains a hosted-device
  requirement. Doctests and warning-clean Rustdoc pass for all three provider
  crates. Exact implementation-head CI passes CUDA run `30678625029`, ROCm
  run `30678625038`, WGPU run `30678625043`, and macOS Metal run
  `30678625035` at `d36a992`.
- Status: done 2026-07-31; final docs-only closeout-head CI remains the merge
  gate for PR #169.

## HEPH-ROCM-PIVOT-DEVICE-INIT-1 [patch] [perf] — done

- Outcome: initialize native ROCm column-pivoted QR's `Q` identity and
  permutation plus complete-pivoted LU's row/column permutations directly in
  device storage through their existing validation dispatches.
- Scope: ROCm `decomposition/{col_piv_qr,full_piv_lu}.rs`, focused native
  decomposition contracts, and synchronized PM/release records. WGPU/CUDA
  host-backed decomposition replacement and algorithm changes are non-goals.
- Acceptance: every uninitialized output element is assigned before the first
  factorization step; pivoted values, permutations, rank, empty behavior, and
  non-finite rejection remain exact; no extra initialization kernel launch is
  added; warning-denied focused gates and exact-head ROCm/Metal CI pass.
- Risk/change class: `[patch]` internal initialization placement. QR host peak
  identity storage and transfer volume fall from `O(rows²)` to `O(1)`;
  permutation host storage and transfer fall from `O(n)` to `O(1)`. Device
  allocation and decomposition arithmetic are unchanged.
- Evidence: source contracts pin the validation-kernel ABI and prove complete
  `Q`, permutation, and rank assignment. Focused decomposition-feature
  Nextest executes 2/2 source contracts; 4 device-value contracts compile and
  adapterless-skip pending hosted ROCm execution. ROCm all-target check,
  scope-local warning-denied Clippy, doctests, and Rustdoc pass.
- Status: done 2026-07-31. Delivered by PR #170, merge commit `548c181`.
  Exact closeout head `015d82e` passes CUDA run `30680144377`, ROCm run
  `30680144380`, WGPU run `30680144391`, and macOS Metal run `30680144423`.

## HEPH-WGPU-IDENTITY-UNIFORM-PACK-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-identity-uniform-pack`; scope: WGPU matrix
  identity dispatch metadata, inherited Metal behavior, focused matrix-power
  contracts, and synchronized PM evidence.
- Outcome: place trait-defined additive and multiplicative identity values in
  one pooled allocation with separately aligned ranges so each non-empty
  identity dispatch acquires two uniform buffers rather than three.
- Non-goals: matrix-power arithmetic, kernel launch count, output allocation,
  scalar identity semantics, CUDA/ROCm launch ABIs, or unmeasured runtime
  claims.
- Acceptance: custom nonstandard scalar identities, admitted vector token
  layouts, and built-in integer/float identities remain exact; empty and
  non-square behavior is unchanged; zero and one ranges satisfy device uniform
  offset alignment without host padding allocation; warning-denied WGPU gates
  and exact-head WGPU/macOS Metal CI pass.
- Risk/change class: `[patch]` internal uniform pooling. Per-dispatch uniform
  acquisitions fall from three to two by construction; queue writes, bindings,
  output storage, and peak matrix memory are unchanged.
- Evidence: shader source contracts preserve separate typed identity bindings
  for scalar and vector tokens, while the buffer-layout contract pins a
  device-aligned second range for three-lane values. Local WGPU execution passes
  the nonstandard-identity,
  built-in exponent-zero, empty, odd-power, strided, and non-square matrix-power
  contracts. Formatting, all-target check, warning-denied Clippy, doctests, and
  Rustdoc pass offline. The Atlas development overlay changes the unused-patch
  lock set, so local `--locked` commands stop before compilation; exact-head
  hosted CI remains the clean-lock evidence.
- Status: done 2026-07-31. Delivered by PR #171, merge commit `1710f36`.
  Final docs head `eec0f03` passes CUDA run `30682601670`, ROCm run
  `30682601650`, WGPU run `30682601688`, and macOS Metal run `30682601655`.
  Implementation and focused local gates complete 2026-07-31.
  Independent review approves the general scalar/vector layout and evidence
  boundary. Exact implementation head `106a475` passes CUDA run `30682144677`,
  ROCm run `30682144681`, WGPU run `30682144674`, and macOS Metal run
  `30682144672`.

## HEPH-WGPU-MATRIX-PROPERTIES-HOST-REUSE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-matrix-properties-host-reuse`; scope:
  WGPU host-delegated rank/determinant scratch ownership, focused layout/value
  contracts, and synchronized PM evidence.
- Outcome: reuse the downloaded backing buffer directly as mutable Gaussian-
  elimination scratch when the matrix layout is canonical C-contiguous at
  offset zero, avoiding a second logical-matrix allocation and copy.
- Non-goals: rank/determinant arithmetic, device download/upload behavior,
  strided-view compaction, CUDA/ROCm implementations, public APIs, or
  unmeasured runtime claims.
- Acceptance: the contiguous callback receives the downloaded buffer prefix
  in place; strided layouts still compact into independent row-major storage;
  focused rank/determinant values remain unchanged; formatting, all-target
  check, warning-denied Clippy, Nextest, doctests, and Rustdoc pass.
- Risk/change class: `[patch]` internal ownership optimization. Canonical
  contiguous inputs eliminate one `rows * cols` host allocation and copy by
  construction; the full backing-buffer download and determinant upload remain.
- Evidence: two pointer/value ownership contracts and four existing
  rank/determinant device-value contracts pass focused Nextest. Formatting,
  all-target check, warning-denied Clippy, doctests, and Rustdoc complete
  locally. Initial exact-head WGPU run `30707193470` and Metal run
  `30707193482` exposed a pre-existing base-head feature-gating defect:
  decomposition seam modules imported feature-gated decomposition modules in
  no-default-feature builds. WGPU and Metal now gate the seam modules and
  re-exports consistently. Refreshed Metal run `30707678624` then exposed the
  same missing feature gate on its macOS-only decomposition conformance target;
  that target now shares the capability gate. Exact WGPU no-default library and
  Metal no-default all-target checks pass locally. Rustdoc reports a
  pre-existing broken `StencilOps` link in
  peer-owned stencil scope. The broad pre-edit Nextest command exceeded the
  120-second shell ceiling during compilation, and a broad post-edit attempt
  failed linking the unrelated `convolution_contracts` binary before tests;
  narrowed affected-target runs pass.
- Status: implementation and focused local gates complete 2026-08-01.
  Independent review approves the scratch-reuse and symmetric module/re-export
  feature-gating diffs after excluding generated Atlas overlay lockfile
  reordering, and separately approves the Metal test-target feature gate.
  Exact implementation head `1169f99` passes CUDA run `30708041475`, ROCm run
  `30708041480`, WGPU run `30708041490`, and macOS Metal run `30708041462`.
  During docs-only closeout CI, `master` advanced to `275f622`; GitHub's merge
  ref exposed its Metal f64 convolution contract importing `ComputeDevice`
  instead of the owning `ComputeDeviceCapabilities` trait. Cross-target
  reproduction then proved the deeper provider gap: `MetalDevice` did not
  implement that shared seam. The branch now integrates the base, adds the real
  Metal capability delegation, and uses the owning trait in the contract. The
  macOS cross-target contract build, focused value-semantic capability test,
  Metal all-target check, and warning-denied clippy pass locally; independent
  review approves the complete limits and five-feature delegation coverage.
  Integrated implementation head `bfe934f` passes CUDA run `30708999594`, ROCm
  run `30708999598`, WGPU run `30708999600`, and macOS Metal run `30708999596`.
  The final docs-only closeout-head matrix remains PR #172's merge gate.

## HEPH-METAL-VOLUME-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-metal-volume-overwrite`; scope: Metal's
  allocating ray-line-integral wrapper, focused analytical contracts, and
  synchronized PM evidence.
- Outcome: allocate the private ray-count-sized output through the
  overwrite-before-read seam before the delegated WGPU kernel assigns every
  validated ray result.
- Non-goals: ray traversal arithmetic, field/ray layouts, caller-owned output,
  WGPU/CUDA/ROCm implementations, benchmark workloads, and runtime claims
  without matched measurements.
- Acceptance: validation precedes allocation; every non-empty output element
  is assigned before exposure; empty output remains valid; analytical Metal
  ray-integral contracts and warning-denied provider gates pass; exact-head
  backend CI passes.
- Risk/change class: `[patch]` internal Metal allocation policy. The change
  removes one redundant ray-count-sized logical initialization while WebGPU's
  mandatory platform initialization behavior remains implementation-defined.
- Evidence: source review proves validation precedes allocation and the
  delegated one-work-item-per-ray kernel assigns hit, miss, and grazing paths;
  empty batches expose no readable element. Local format, all-target check,
  warning-denied Clippy, and focused Nextest pass against the Atlas overlay.
  The four local Metal tests return through the documented no-adapter path on
  Windows. Exact head `f857c84` passes CUDA run `30674702291`, ROCm run
  `30674702372`, WGPU run `30674702296`, and native macOS Metal run
  `30674702295`. The first native Metal run exposed an invalid bitwise norm
  oracle; the corrected shared contract bounds the backend square root to one
  output ULP while retaining exact arithmetic assertions elsewhere.
- Status: done 2026-07-31. Delivered by PR #168, merge commit `b7ff88b`.

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

## HEPH-ROCM-PIVOT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-rocm-pivot-overwrite`; scope: ROCm native
  complete-pivot LU and column-pivot QR factor-matrix startup copies, focused
  Leto-differential contracts, and synchronized PM evidence.
- Outcome: allocate each non-empty private factor matrix through the
  overwrite-before-read seam before the validated dense copy or strided
  identity kernel assigns every logical element.
- Non-goals: decomposition arithmetic, pivot/status/threshold initialization,
  empty outputs, WGPU/CUDA host-upload implementations, allocation capacities,
  benchmark workloads, and runtime claims without matched measurements.
- Acceptance: both private factor allocations are exposed only after their
  complete copy dispatch and decomposition stages succeed; dense and strided
  inputs preserve Leto values, pivots, rank, solve, and inverse/least-squares
  contracts; warning-denied ROCm gates and exact-head backend CI pass.
- Risk/change class: `[patch]` internal ROCm allocation policy. Each non-empty
  successful factorization omits one initialization transfer of the complete
  factor matrix. Peak allocation, decomposition order, and public ownership are
  unchanged.
- Evidence: focused decomposition-feature nextest contracts pass for
  complete-pivot LU (3/3) and column-pivot QR (3/3), including Leto values,
  dense/strided layouts, ranks, solves, inverse, least-squares, and non-finite
  rejection. Warning-denied ROCm clippy passes. Independent review approves
  complete write coverage, private failure paths, stream ordering, empty
  behavior, and the bounded transfer claim with high confidence.
  Implementation-head CI passes on CUDA (`91147530954`), ROCm (`91147531264`),
  WGPU (`91147531030`), and macOS Metal (`91147530759`).
- Status: done 2026-07-31. Delivered by PR #163.

## HEPH-AXIS-REDUCTION-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-axis-reduction-overwrite`; scope: WGPU,
  CUDA, and ROCm allocating rank-2 sum/product/min/max/mean outputs, with Metal
  inheriting WGPU; exact axis contracts and synchronized PM evidence.
- Outcome: remove redundant initialization before each allocating axis kernel
  assigns every logical output element exactly once.
- Non-goals: caller-owned `*_into` buffers, prepared-axis output ownership,
  axis arithmetic or layouts, empty-axis semantics, allocation capacities,
  benchmark workloads, and runtime claims without matched measurements.
- Acceptance: overwrite-before-read storage is returned only after axis, shape,
  layout, alias, and width validation plus successful dispatch; invalid calls
  drop their private allocation. Generic and mean outputs preserve Leto values
  for non-empty, empty, strided, and beyond-block-width cases;
  warning-denied WGPU/CUDA/ROCm gates and exact-head WGPU/CUDA/ROCm/macOS-Metal
  CI pass.
- Risk/change class: `[patch]` internal allocation policy. CUDA and ROCm omit
  one initialization transfer of `output_len * size_of::<T>()` bytes per
  non-empty successful allocating axis-reduction call. WGPU and Metal record
  the same overwrite contract while WebGPU retains its mandated initialization
  behavior. Peak allocation and zero-length output behavior remain unchanged.
- Evidence: exact-base provider CI on `b565f25` supplies the unchanged generic,
  mean, strided, empty-axis, and beyond-block-width semantic baseline. Kernel
  review proves each dispatched logical index computes one validated output
  offset and assigns it directly; zero-length outputs return no readable
  elements. A residue scan leaves only WGPU's initialized one-element dummy
  binding for empty input storage. Independent review approves write coverage,
  validation/error ordering, asynchronous queue ordering, zero-length behavior,
  and the bounded transfer claim. Focused local Leto-differential contracts pass
  for WGPU (1/1), CUDA (1/1), and ROCm (1/1). Implementation-head CI passes on
  WGPU (`91142020221`), CUDA (`91142020156`), ROCm (`91142020205`), and macOS
  Metal (`91142019842`).
- Status: done 2026-07-31. Delivered by PR #162.

## HEPH-SCALAR-REDUCTION-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-reduction-overwrite`; scope: WGPU, CUDA,
  and ROCm scalar-reduction tree and prepared-plan outputs, with Metal
  inheriting WGPU; exact reduction contracts and synchronized PM evidence.
- Outcome: remove redundant output initialization before immediate singleton
  copies and reduction kernels overwrite every immediate or private
  intermediate result.
- Non-goals: empty-input identity uploads, reduction arithmetic or order, axis
  reductions, allocation capacities, benchmark workloads, and runtime claims
  without matched measurements.
- Acceptance: overwrite-before-read storage is used only for non-empty scalar
  reductions; generic sum/min/max and width-boundary contracts pass on WGPU and
  physical CUDA; prepared reuse and batching preserve exact outputs;
  warning-denied WGPU/CUDA/ROCm gates and exact-head WGPU/CUDA/ROCm/macOS-Metal
  CI pass.
- Risk/change class: `[patch]` internal allocation policy. CUDA and ROCm avoid
  one initialization transfer per immediate reduction-tree pass and private
  prepared intermediate; CUDA also avoids the singleton output initialization.
  Prepared final outputs remain initialized because `output()` is callable
  before dispatch. WGPU and Metal record the same overwrite contract while
  WebGPU retains its mandated initialization behavior. Peak allocation and
  empty-input identities remain unchanged.
- Evidence: exact-base provider CI on `ca2b875` supplies the unchanged semantic
  baseline. Kernel review proves thread zero in each launched workgroup writes
  its unique partial, singleton copies overwrite their sole output, and each
  prepared private intermediate is consumed only by the following pass. CUDA
  and ROCm contracts now assert the public final output is zero before first
  dispatch. Independent review approves the corrected final/intermediate
  boundary. Local compilation is blocked by concurrent uncommitted Leto zip
  changes with 47 lifetime/privacy diagnostics; clean-checkout CI is the
  executable gate.
- Hosted evidence on implementation head `476bc77`: WGPU job `91137591572`
  passes in 6m43s, CUDA job `91137591608` in 6m17s, ROCm job `91137591201` in
  5m53s, and macOS Metal job `91137591647` in 6m54s. Required-device NVIDIA
  and AMD jobs skip because dedicated runners are unavailable; no physical
  CUDA or ROCm execution claim is made for this increment.
- Status: done 2026-07-31; delivered through PR #161. Claimed files: WGPU,
  CUDA, and ROCm scalar
  reduction implementations and prepared plans; reduction contracts;
  `CHANGELOG.md`, `backlog.md`, and `checklist.md`.

## HEPH-WGPU-QR-DIRECT-EIGHT-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-qr-direct-eight`; scope: WGPU blocked-QR
  direct-route threshold, its routing-boundary value contracts, matched
  `blocked_qr_tail` measurements, and synchronized PM evidence. Metal inherits
  the WGPU implementation.
- Outcome: route matrices spanning at most eight 32-column panels through one
  dense readback and host factorization only when matched Criterion evidence
  proves this faster than the retained panel-blocked path.
- Non-goals: decomposition arithmetic, the retained blocked algorithm, CUDA or
  ROCm implementations, benchmark workload changes, and performance claims
  beyond the measured shapes.
- Acceptance: 256-column direct and 257-column blocked boundary cases preserve
  complete factorization, reconstruction, and solve contracts; 192x129 and
  384x256 benchmark change intervals are strictly negative at 95% confidence;
  the unchanged 192x128 control does not shift significantly; warning-denied
  WGPU gates and exact-head WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal routing policy. At 384x256 the direct
  route removes about 98,816 bytes of persistent blocked-path device scratch
  and six mapping polls, but replaces paired 98,304-byte compact staging with
  one transient 393,216-byte dense readback. Reject the change if the matched
  measurements or boundary contracts fail.
- Evidence: the fresh four-panel baseline measures 192x128 at 354.91–359.10 us,
  192x129 at 937.57–952.44 us, and 384x256 at 2.6701–2.7027 ms. Raising the
  direct limit to eight panels leaves the 192x128 control unchanged
  (`p = 0.06`), improves 192x129 by 62.750–63.864%, but regresses 384x256 by
  707.13–718.54%. The candidate therefore fails its required strictly-negative
  target intervals.
- Status: done 2026-07-31. The eight-panel hypothesis is rejected and the
  four-panel route is retained; no runtime source or benchmark change ships.

## HEPH-DECOMPOSITION-WORKSPACE-OVERWRITE-1 [patch] — done

- Owner: Codex on `codex/hephaestus-decomposition-workspace`; scope: reusable
  blocked Cholesky, LU, and QR panel-transfer workspaces in WGPU and CUDA, with
  Metal inheriting WGPU.
- Outcome: remove redundant initialization before each workspace's active
  prefix is overwritten by a region-copy kernel or host upload.
- Non-goals: decomposition arithmetic, startup-copy destinations, ROCm's
  native whole-device algorithms, workspace capacities, benchmark workloads,
  or runtime claims without matched measurements.
- Acceptance: overwrite-before-read storage is used only where every consumed
  element is first written; exact blocked-decomposition value contracts pass
  on WGPU and physical CUDA; warning-denied provider gates and exact-head
  WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal allocation policy. CUDA avoids two
  initialization transfers per blocked QR call. WGPU and Metal record the
  overwrite contract while preserving WebGPU's platform-managed
  initialization. Peak allocation is unchanged.
- Evidence: WGPU blocked-decomposition contracts pass 19/19 before and after
  the change; physical CUDA contracts pass 18/18 before and after. Rust 1.97
  package formatting and warning-denied all-target Clippy pass for WGPU and
  feature-enabled CUDA. Region gathers and host uploads overwrite every active
  compact-panel, reflector-vector, and reflector-metadata prefix before the
  corresponding copy or update consumes it.
- Hosted evidence on implementation head `be734be`: WGPU job `91084690146`
  passes in 6m41s, CUDA job `91084690094` in 6m09s, ROCm job `91084735821` in
  6m13s, and macOS Metal job `91084744715` in 7m08s. Required-device NVIDIA
  and AMD jobs skip because dedicated runners are unavailable; physical CUDA
  value contracts pass locally, and no physical ROCm execution claim is made.
- Status: done 2026-07-31; delivered through PR #160. Claimed files: WGPU
  blocked Cholesky, LU, and QR implementations; CUDA blocked QR;
  `CHANGELOG.md`, `backlog.md`, and `checklist.md`.

## HEPH-MATRIX-PROPERTIES-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-matrix-properties-overwrite`; scope: CUDA
  and ROCm matrix-rank/determinant scratch, rank, and determinant allocations.
- Outcome: remove six redundant initialization transfers before each native
  single-thread matrix-properties kernel copies every scratch element and
  writes both scalar outputs.
- Non-goals: WGPU's host-delegated matrix-properties path, row-reduction
  arithmetic, tolerance semantics, layouts, benchmark workloads, or runtime
  claims without matched measurements.
- Acceptance: CUDA and ROCm use overwrite-before-read storage only after
  validating non-empty dimensions, tolerance, layout, and launch widths;
  rank/determinant value and tolerance contracts pass on physical CUDA;
  warning-denied provider gates and exact-head WGPU/CUDA/ROCm/macOS-Metal CI
  pass.
- Risk/change class: `[patch]` internal allocation policy. CUDA and ROCm avoid
  one matrix-sized scratch initialization and two scalar initializations per
  matrix-properties call. Peak allocation and WGPU/Metal behavior are
  unchanged.
- Evidence: physical CUDA rank and determinant contracts pass 2/2 before and
  after the allocation change, including exact full-rank, rank-deficient,
  singular, rectangular, and tolerance-discriminator values. Rust 1.95
  formatting and warning-denied all-target Clippy pass for feature-enabled
  CUDA and adapterless ROCm. The native kernel copies all `rows * cols`
  scratch elements before elimination and assigns both scalar outputs on every
  path.
- Hosted evidence on implementation head `f57180e`: WGPU job `90814481300`
  passes in 6m32s, CUDA job `90814481682` in 7m47s, ROCm job `90814482979`
  in 5m50s, and macOS Metal job `90814482965` in 6m22s. Required-device
  NVIDIA job `90814482224` and AMD job `90814483598` skip because hosted
  hardware runners are unavailable; physical CUDA value contracts pass
  locally, and no physical ROCm execution claim is made.
- Status: done 2026-07-30; delivered through PR #158. Claimed files:
  CUDA/ROCm matrix-rank implementations and contracts, `CHANGELOG.md`,
  `backlog.md`, and `checklist.md`.

## HEPH-MAP-REDUCTION-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-map-reduction-overwrite`; scope: fused
  map-reduction first-pass partials and L2 square-root outputs in WGPU, CUDA,
  and ROCm, with Metal inherited through WGPU.
- Outcome: remove redundant initialization transfers before kernels overwrite
  every non-empty map-reduction partial and L2 result.
- Non-goals: reduction-tree identity buffers, empty-input identities, operation
  arithmetic, layouts, benchmark workloads, or runtime claims without matched
  measurements.
- Acceptance: all six non-empty allocations use overwrite-before-read storage;
  dot, trace, L1/L2/max norm, prepared reuse, strided, reversed-view, and empty
  identity contracts retain exact values; warning-denied provider gates and
  exact-head WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal allocation policy. CUDA and ROCm avoid
  one first-pass partial initialization transfer per plan and one scalar
  initialization transfer per L2 plan; WGPU and Metal preserve
  platform-managed initialization behavior. Peak allocation is unchanged.
- Evidence: WGPU and physical CUDA value contracts pass 3/3 before and after
  the allocation change, covering exact norms, prepared resource reuse and
  input mutation, strided/reversed layouts, and empty identities. Rust 1.95
  formatting and warning-denied all-target Clippy pass for WGPU,
  feature-enabled CUDA, and adapterless ROCm. Every non-empty first-pass
  workgroup stores its indexed partial, and the unary launch stores the sole
  L2 output.
- Hosted evidence on implementation head `561d396`: WGPU job `90810083486`
  passes in 4m46s, CUDA job `90810110166` in 7m36s, ROCm job `90810111786`
  in 5m54s, and macOS Metal job `90810090280` in 7m21s. Required-device
  NVIDIA job `90810110823` and AMD job `90810112632` skip because hosted
  hardware runners are unavailable; physical CUDA value contracts pass
  locally, and no physical ROCm execution claim is made.
- Status: done 2026-07-30; delivered through PR #157. Claimed files: WGPU map
  reduction, CUDA/ROCm norm implementations and contracts, `CHANGELOG.md`,
  `backlog.md`, and `checklist.md`.

## HEPH-VOLUME-OUTPUT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-volume-overwrite`; scope: allocating ray
  line-integral outputs in WGPU, CUDA, and ROCm, with Metal inherited through
  WGPU, plus exact analytical value contracts.
- Outcome: remove redundant initialization of one ray-count-sized output
  before the volume kernel writes one integral for every validated ray.
- Non-goals: ray traversal arithmetic, field/ray layouts, caller-owned output
  semantics, reduction buffers, benchmark workloads, or runtime claims without
  matched measurements.
- Acceptance: all three allocating wrappers use uninitialized storage only
  after full launch/input validation; analytical ray-integral contracts pass on
  WGPU and physical CUDA; ROCm retains feature-enabled hosted coverage;
  warning-denied provider gates and exact-head WGPU/CUDA/ROCm/macOS-Metal CI
  pass.
- Risk/change class: `[patch]` internal allocation policy and one avoided
  ray-count-sized initialization transfer on CUDA/ROCm; peak allocation is
  unchanged.
- Evidence: WGPU analytical contracts pass 4/4 and physical CUDA contracts
  pass 3/3 before and after the allocation change. Rust 1.95 warning-denied
  all-target Clippy passes for WGPU, feature-enabled CUDA, and adapterless
  ROCm. The validated one-work-item-per-ray launch writes every output slot.
- Hosted evidence on implementation head `5c32355`: WGPU job `90801212471`,
  CUDA job `90801212567`, ROCm job `90801212912`, and macOS Metal job
  `90801212565` pass. Required-device NVIDIA job `90801213164` and AMD job
  `90801213519` skip because hosted hardware runners are unavailable; physical
  CUDA value contracts pass locally, and no physical ROCm execution claim is
  made.
- Status: done 2026-07-30; delivered through PR #156. Claimed files:
  WGPU/CUDA/ROCm volume implementations and contracts, `CHANGELOG.md`,
  `backlog.md`, and `checklist.md`.
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

## HEPH-DECOMPOSITION-STARTUP-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-decomposition-overwrite`; scope: the
  full-matrix startup-copy destinations for blocked Cholesky, LU, and QR in
  WGPU, CUDA, and ROCm, with Metal inherited through WGPU.
- Outcome: remove redundant initialization of nine matrix-sized destinations
  before whole-buffer copies or canonical strided identity kernels fully
  overwrite them.
- Non-goals: panel, reflector, status, rank, threshold, or metadata buffers;
  decomposition arithmetic; blocking strategy; layout contracts; benchmark
  workloads; or runtime claims without matched measurements.
- Acceptance: only the nine full-matrix startup-copy destinations use
  uninitialized storage; full-write proofs cover dense copies and strided
  identity dispatches; existing value-semantic blocked decomposition contracts
  pass on WGPU and physical CUDA, ROCm contracts retain compile/value coverage,
  warning-denied provider gates pass, and exact-head WGPU/CUDA/ROCm/macOS-Metal
  CI is green.
- Risk/change class: `[patch]` internal allocation policy and one avoided
  matrix-sized initialization transfer per decomposition call on CUDA/ROCm;
  peak allocation is unchanged.
- Evidence: WGPU blocked-decomposition contracts pass 17/17 and physical CUDA
  contracts pass 16/16 before and after the allocation change. Rust 1.95
  warning-denied all-target Clippy passes for WGPU, feature-enabled CUDA, and
  adapterless ROCm. Each dense whole-buffer copy and strided identity dispatch
  covers the destination's validated matrix extent before decomposition reads.
  Adapterless ROCm exposes no decomposition tests, so feature-enabled ROCm
  behavioral coverage remains an explicit hosted-CI requirement.
- Hosted evidence on implementation head `8916a22`: WGPU job `90797891213`
  passes in 6m27s, CUDA job `90797891064` in 6m25s, feature-enabled ROCm job
  `90797891293` in 5m57s, and macOS Metal job `90797891343` in 5m43s.
  Required-device NVIDIA and AMD jobs skip because no hardware runners are
  configured; local physical CUDA supplies the available device evidence.
- Status: done 2026-07-30. Delivered by PR #155.

## HEPH-MATPOW-OUTPUT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-matpow-overwrite`; scope: matrix-power
  base-copy and multiply-scratch allocations in WGPU, CUDA, and ROCm, with
  Metal inherited through WGPU, plus exact allocated-output contracts.
- Outcome: remove redundant initialization of three matrix-sized buffers per
  matrix-power call before canonical strided-copy and matrix-multiply kernels
  fully overwrite them.
- Non-goals: exponentiation order, matrix-multiply kernels, caller-owned
  outputs, scalar precision, layout semantics, benchmark workloads, or runtime
  claims without matched measurements.
- Acceptance: all nine base/scratch allocations use uninitialized storage only
  after square-shape and allocation-size validation; CUDA gains the missing
  value-semantic matrix-power contract; existing WGPU/ROCm contracts and the
  new CUDA contract exercise identity, odd powers, strided input, and shape
  rejection; warning-denied package gates and exact-head
  WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal allocation policy and up to three
  avoided matrix-sized initialization transfers on CUDA/ROCm; peak allocation
  is unchanged.
- Evidence: WGPU matrix-power contracts pass 2/2; the physical-CUDA Leto,
  identity, strided-input, and non-square contract passes 1/1; the adapterless
  ROCm contract passes 1/1 as compile/contract coverage. Rust 1.95
  warning-denied all-target Clippy passes for WGPU, feature-enabled CUDA, and
  ROCm. The canonical strided identity kernel writes every base element and
  `matmul_into` writes every scratch element before either buffer is read.
- Hosted evidence on implementation head `2635fe7`: WGPU job `90794414537`
  passes in 6m31s, CUDA job `90794414508` in 7m03s, ROCm job `90794414660`
  in 6m05s, and macOS Metal job `90794414557` in 5m58s. Required-device
  NVIDIA and AMD jobs skip because no hardware runners are configured; local
  physical CUDA supplies the available device evidence.
- Status: done 2026-07-30. Delivered by PR #154.

## HEPH-SPARSE-OUTPUT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-sparse-overwrite`; scope: allocating
  sparse matrix-vector and sparse matrix-dense-matrix wrappers in WGPU, CUDA,
  and ROCm, with Metal inherited through WGPU, plus exact allocated-output
  contracts.
- Outcome: remove redundant device-output initialization before sparse kernels
  fully overwrite their contiguous results.
- Non-goals: caller-owned outputs, prepared-plan allocation, CSR validation,
  arithmetic order, sparse formats, benchmark workloads, or runtime claims
  without matched measurements.
- Acceptance: all six allocating wrappers use uninitialized storage only after
  output-shape validation; exact allocated SpMV and SpMM contracts cover empty
  rows and non-empty products on applicable devices; warning-denied package
  gates and exact-head WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal allocation policy and one avoided
  output-sized initialization transfer on CUDA/ROCm; peak allocation is
  unchanged.
- Status: done 2026-07-30. Implementation head `51ffe32` passes exact WGPU and
  physical CUDA sparse contracts, adapterless ROCm contract
  compilation/execution, Rust 1.95 warning-denied WGPU/CUDA/ROCm all-target
  Clippy, formatting, and diff checks. PR #153 head `3499245` passed WGPU job
  `90789708535`, CUDA job `90789708499`, ROCm job `90789708424`, and macOS
  Metal job `90789709223`.

## HEPH-LINALG-OUTPUT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-linalg-overwrite`; scope: allocating
  Kronecker-product, matrix-multiply, and batched-matrix-multiply wrappers in
  WGPU, CUDA, and ROCm, with Metal inherited through WGPU, plus exact
  allocated-output contracts.
- Outcome: remove redundant device-output initialization before linalg kernels
  fully overwrite their contiguous results.
- Non-goals: caller-owned outputs, tiling, accumulation order, scalar
  precision, layout semantics, benchmark workloads, or runtime claims without
  matched measurements.
- Acceptance: all nine allocating wrappers use uninitialized storage only
  after output-layout validation; exact allocated contracts cover the three
  operation families on applicable devices; warning-denied package gates and
  exact-head WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal allocation policy and one avoided
  output-sized initialization transfer on CUDA/ROCm; peak allocation is
  unchanged.
- Status: done. Implementation head `903e8b5` passed CUDA job `90746818443`,
  ROCm job `90746818452`, WGPU job `90746818469`, and macOS Metal job
  `90746818550`.

## HEPH-STRIDED-OUTPUT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-strided-overwrite`; scope: allocating
  binary, typed-binary, unary, and scalar strided elementwise wrappers in
  WGPU, CUDA, and ROCm, with Metal inherited through WGPU, plus exact
  allocated-output contracts.
- Outcome: remove redundant device-output initialization before strided
  elementwise kernels fully overwrite their contiguous result.
- Non-goals: caller-owned outputs, dynamic-rank dispatch, operation formulas,
  layout semantics, benchmark workloads, or runtime claims without matched
  measurements.
- Acceptance: all twelve allocating wrappers use uninitialized storage only
  after output-layout validation; allocated CUDA and ROCm contracts cover all
  four operation forms and empty results; existing WGPU allocated contracts
  remain exact; warning-denied package gates and exact-head
  WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[patch]` internal allocation policy and one avoided
  output-sized initialization transfer on CUDA/ROCm; peak allocation is
  unchanged.
- Status: done. Implementation head `8ca789c` passed CUDA job `90742542272`,
  ROCm job `90742542268`, WGPU job `90742542220`, and macOS Metal job
  `90742542356`.

## HEPH-SCAN-OUTPUT-OVERWRITE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-scan-overwrite`; scope: allocating
  rank-2 scan wrappers in WGPU, CUDA, and ROCm, with Metal inherited through
  WGPU, plus exact allocated-output contracts and synchronized evidence.
- Outcome: remove redundant device-output initialization before scan kernels
  fully overwrite the contiguous result.
- Non-goals: scan arithmetic, launch geometry, caller-owned outputs, empty
  allocation semantics, benchmark workloads, or runtime claims without matched
  measurements.
- Acceptance: non-empty allocating cumulative sum/product and suffix
  sum/product paths allocate uninitialized storage only after layout validation;
  exact allocated-output Leto contracts pass on applicable local devices;
  warning-denied package gates and exact-head WGPU/CUDA/ROCm/macOS-Metal CI
  pass.
- Risk/change class: `[patch]` internal allocation policy and one avoided
  output-sized device initialization transfer; peak allocation is unchanged.
- Status: done 2026-07-29. Rust 1.95 warning-denied WGPU, CUDA, and ROCm
  package gates pass; WGPU scan Nextest passes 2/2 and physical CUDA scan
  Nextest passes 4/4. Exact implementation head `ae9d440` passed WGPU
  `90714878716`, CUDA `90714887588`, ROCm `90714879225`, and macOS Metal
  `90714878623`; hardware-only jobs skipped without selected device runners.

## HEPH-COEUS-ACTIVATION-PARITY-RECONCILE-1 [patch] — done

- Owner: Codex on `codex/hephaestus-coeus-parity-reconcile`; scope:
  Hephaestus activation-parity checklist state and the already-merged Coeus
  consumer evidence.
- Outcome: reconcile the provider checklist with direct Coeus ROCm/Metal
  routing, Leto CPU differential tests, and exact-head WGPU/CUDA/ROCm/Metal CI.
- Non-goals: provider source changes, new operation contracts, release/version
  transitions, or runtime and memory claims.
- Acceptance: every stale activation, exact-GELU, unary-math, and
  activation-tail consumer item cites its merged Coeus PR and exact CI jobs;
  source inspection confirms no local fallback or duplicated formula.
- Risk/change class: `[patch]` tracking reconciliation only.
- Status: done 2026-07-29. Source inspection confirms Coeus ROCm and Metal
  dispatch the shared activation markers directly and derive expected values
  through `coeus_leto`; Coeus PRs #223, #226, #230, and #237 provide
  exact-head green WGPU/CUDA/ROCm/Metal consumer evidence. Required-device
  ROCm jobs remained runner-gated.

## HEPH-WGPU-QR-REGION-TRANSFER-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-qr-wide-transfer`; scope: the retained WGPU
  blocked-QR path at and above the 129-column routing boundary.
- Governing decision:
  [`docs/adr/0038-blocked-qr-final-panel-synchronization.md`](docs/adr/0038-blocked-qr-final-panel-synchronization.md).
- Outcome: identify and remove measured per-panel region-transfer resource
  construction or synchronization overhead while preserving the hybrid GPU
  trailing-update algorithm.
- Non-goals: widening the direct-host threshold without matched evidence,
  changing QR arithmetic or block width, backend algorithm changes, weakening
  workloads, or performance and memory claims without matched measurements.
- Acceptance: a value-validating production profile covers 128/129 columns and
  at least one wider multi-panel shape; the selected increment reduces a
  measured runtime or live-allocation bound, preserves complete Leto `R`, solve,
  and reconstruction contracts, and passes warning-denied package gates plus
  exact-head WGPU/CUDA/ROCm/macOS-Metal CI.
- Risk/change class: `[patch]` internal transfer orchestration and memory reuse.
- Status: done 2026-07-29. The retained panel workspace removes two download
  preparations at 192×129 and five at 384×256 without increasing peak staging;
  local package gates and exact-head WGPU/CUDA/ROCm/macOS-Metal CI pass.
  Hardware CUDA and ROCm jobs remain runner-gated.

## HEPH-WGPU-QR-WIDE-PROFILE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-qr-wide-profile`; scope: WGPU blocked-QR
  matrices wider than two 32-column panels, component profiling, and one
  evidence-selected production increment.
- Outcome: identify the latency and live-memory bound in the remaining wide
  blocked path after PR #143 removed narrow-only orchestration, then optimize
  that bound without changing QR arithmetic or the measured workload.
- Non-goals: benchmark-instrument tuning, speculative block-width changes,
  silent CPU fallback for wide tails, CUDA/ROCm/Metal algorithm changes, or
  runtime and memory claims without matched evidence.
- Acceptance: a value-validating component profile covers the 64/65-column
  legacy boundary and active 128/129-column routing boundary; the selected
  increment preserves complete Leto `R`, solve, and reconstruction contracts,
  records a matched Criterion A/B result and live-allocation model, passes
  warning-denied package gates, and passes exact-head
  WGPU/CUDA/ROCm/macOS-Metal CI.
- Risk/change class: `[patch]` performance audit until evidence selects a
  behavior or public-contract change.
- Status: done 2026-07-29; implementation, matched local evidence, package
  verification, review-requested 129-column profile coverage, and exact-head
  WGPU/CUDA/ROCm/macOS-Metal CI complete.

## HEPH-WGPU-QR-POST-TAIL-PROFILE-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-qr-post-tail-profile`; scope: WGPU blocked
  QR component profiling and the next evidence-selected vertical optimization.
- Outcome: re-establish the 70×35 blocked-QR latency decomposition after PR
  #142 removed the final-panel synchronization, then optimize the new binding
  production component without changing arithmetic or benchmark workload.
- Non-goals: benchmark-instrument tuning, block-width changes, CPU fallback for
  wide tails, CUDA/ROCm/Metal algorithm changes without provider evidence, or
  unmeasured runtime and memory claims.
- Acceptance: a controlled component profile validates every factorization,
  identifies the new latency bound, and records the measured allocation and
  synchronization model; the selected production increment has value-semantic
  Leto differential coverage, a matched Criterion A/B result, warning-denied
  checks, and exact-head WGPU/CUDA/ROCm/macOS-Metal CI.
- Risk/change class: `[patch]` performance audit until evidence selects a
  behavior or public-contract change.
- Status: merged in PR #143 at `6a15e17`. Exact review-follow-up head
  `60a1843` passed WGPU job `90494790828`, CUDA `90494790688`, ROCm
  `90494790469`, and macOS Metal `90494791112`; hardware-only jobs skipped.

## HEPH-WGPU-QR-TAIL-SYNC-1 [minor] [perf] — done

- Owner: Codex on `codex/hephaestus-wgpu-qr-tail-sync`; scope:
  `hephaestus-core` packed-panel Householder application, WGPU paired
  matrix-region transfer, the blocked-QR final two-panel path, value-semantic
  contracts, a Criterion before/after benchmark, ADR, and synchronized
  performance artifacts.
- Outcome: finish the final two QR panels after one paired readback, reusing the
  existing panel and compact device buffers so the blocked path removes one
  host/device synchronization without adding persistent compact scratch.
- Non-goals: changing QR block width or arithmetic precision, CPU fallback for
  earlier panels, a second QR algorithm, CUDA/ROCm/Metal decomposition changes,
  benchmark workload reduction, or cross-device speed claims.
- Acceptance: matrices on both sides of the 32-column block boundary preserve
  reconstruction and least-squares contracts; the final two-panel path performs
  one paired readback and no trailing GPU reflector dispatch; device scratch
  remains bounded by the existing two `m * block_size` compact buffers; the
  bounded extra live staging is quantified; a matched Criterion baseline/result
  run validates every factorization; formatting, warning-denied checks, focused
  Nextest, doctests, semver checks, benchmark smoke, and exact-head
  WGPU/CUDA/ROCm/macOS-Metal CI pass.
- Risk/change class: `[minor]` additive backend-neutral packed-panel operation
  plus a WGPU scheduling optimization. ADR 0038 owns the interface and
  synchronization decision.
- Status: implementation, matched local evidence, semver verification, and
  exact-head locked provider CI complete in PR #142. CUDA job 90444185125,
  ROCm job 90444185117, WGPU job 90444185164, and macOS Metal job 90444185226
  pass. Hardware-only jobs correctly skip on the pull-request event.

## HEPH-WGPU-MIXED-REDUCTION-BATCH-1 [minor] [perf] — done

- Owner: Codex on `codex/hephaestus-mixed-reduction-batch`; scope: unified
  submission of independent prepared scalar and axis reductions, exact WGPU
  contracts, comparative host-latency evidence, and synchronized artifacts.
- Outcome: encode both prepared reduction families into one command encoder and
  submit one command buffer while preserving scalar tree-stage dependencies.
- Non-goals: sharing scratch buffers, changing arithmetic order, dependent
  reductions, shader fusion, CPU fallback, and cross-device performance claims.
- Acceptance: mixed scalar/axis batches preserve exact outputs, singleton and
  empty plans retain their contracts, the unchanged two-call baseline and
  unified result workload validate every output, and formatting,
  warning-denied package checks, focused Nextest, doctests, benchmark smoke,
  and exact-head provider CI pass.
- Risk/change class: `[minor]` additive WGPU batching surface.
- Status: complete in merged PR #141. The mixed exact contract, all-target
  package check, and three matched benchmark runs pass. Exact implementation-
  head jobs passed WGPU `90427454090`, CUDA `90427454188`, ROCm `90427454307`,
  and macOS Metal `90427454254`; AMD and NVIDIA hardware-only jobs skipped as
  designed.

## HEPH-WGPU-EMPTY-BATCH-NOOP-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-empty-batch-noop`; scope: all-no-op
  prepared scalar- and axis-reduction batch submission, exact contracts,
  comparative host-latency evidence, and synchronized performance artifacts.
- Outcome: return before command-encoder allocation and queue submission when
  a batch contains no singleton copy or compute dispatch.
- Non-goals: skipping mixed batches with real work, changing empty-reduction
  identities, shader changes, backend API changes, and cross-device claims.
- Acceptance: empty batch slices and all-empty prepared batches preserve exact
  outputs; mixed batches still execute; matched baseline/result measurements,
  focused Nextest, formatting, warning-denied package checks, doctests,
  benchmark smoke, and exact-head provider CI pass.
- Risk/change class: `[patch]` internal no-op fast path.
- Status: complete in PR #140. Three samples reduce the scalar-plus-axis
  call-pair median from 27.794 µs to 46 ns;
  focused scalar and axis Nextest contracts, warning-denied all-target Clippy,
  doctests, formatting, and benchmark execution pass. Exact implementation-head
  jobs passed WGPU `90391984817`, CUDA `90391983972`, ROCm `90391984432`, and
  macOS Metal `90391985267`; AMD and NVIDIA hardware-only jobs were skipped as
  designed.

## HEPH-WGPU-SCALAR-BATCH-PASS-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-scalar-batch-pass`; scope:
  `hephaestus-wgpu` prepared scalar-reduction batch encoding, its differential
  contract, comparative benchmark, and synchronized performance artifacts.
- Outcome: encode independent reduction trees stage-major so a batch opens one
  compute pass per tree depth instead of one pass per tree stage, without
  changing arithmetic order or scratch/output allocation.
- Non-goals: combining dependent tree stages in one pass, shader fusion, CPU
  fallback, backend API changes, and performance claims without a controlled
  before/after run of the unchanged batch workload.
- Acceptance: singleton and multi-pass batches preserve exact integer values;
  the benchmark validates every output and reports matched baseline/result
  measurements; formatting, warning-denied package checks, focused Nextest,
  doctests, benchmark smoke, and exact-head provider CI pass.
- Risk/change class: `[patch]` internal command-encoding optimization.
- Status: complete in PR #139. Three matched samples reduce the
  eight-reduction median from 234.206 µs to 100.918 µs; focused Nextest,
  warning-denied all-target Clippy, doctests, formatting, and benchmark
  execution pass. Exact implementation-head provider jobs passed WGPU
  `90372903175`, CUDA `90372915738`, ROCm `90372903296`, and macOS Metal
  `90372902759`; AMD and NVIDIA hardware-only jobs were skipped as designed.

## HEPH-WGPU-AXIS-BATCH-PASS-1 [patch] [perf] — done

- Owner: Codex on `codex/hephaestus-axis-batch-pass`; scope:
  `hephaestus-wgpu` prepared axis-reduction batch encoding, its focused
  contract, comparative benchmark, and synchronized performance artifacts.
- Outcome: encode independent prepared axis reductions in one WGPU compute
  pass so a batch pays one pass-construction boundary without adding buffers or
  changing any reduction's arithmetic.
- Non-goals: CPU fallback, shader fusion, cross-reduction dependencies,
  backend API changes, and performance claims without a controlled before/after
  run of the unchanged batch workload.
- Acceptance: the existing value-semantic batch contract passes; a mixed
  prepared-operation batch is covered; the comparative harness validates every
  output and reports matched baseline/result measurements; formatting,
  warning-denied package checks, focused Nextest, doctests, and benchmark smoke
  pass.
- Risk/change class: `[patch]` internal command-encoding optimization.
- Status: complete in PR #138. Three matched samples reduce the
  eight-reduction median from 105.092 µs to 32.250 µs; focused Nextest,
  warning-denied all-target Clippy, doctests, formatting, and benchmark
  execution pass. Exact-head provider jobs passed WGPU `90360294443`, CUDA
  `90360306102`, ROCm `90360292050`, and macOS Metal `90360324163`. Local
  MSVC verification remains independently blocked by `backtrace 0.3.76`
  missing COFF/Gimli optional dependencies.

## HEPH-DEVICE-LOCAL-COW-2 [arch] [minor] [perf] — done

- Owner: Codex on `codex/uninitialized-device-copy`; scope: the shared
  `ComputeDevice` allocation contract and the provider implementations used by
  device-local Coeus storage detachment.
- Outcome: expose an explicit overwrite-before-read allocation operation so a
  consumer that writes every element exactly once does not pay a redundant
  device-wide zero-initialization pass. CUDA and ROCm allocate without memset;
  WGPU and Metal preserve their platform allocation semantics through the same
  seam.
- Non-goals: uninitialized reads, asynchronous allocation, CPU fallback,
  provider-specific consumer APIs, and runtime speed or memory claims without
  controlled measurements.
- Acceptance: all shipped providers and unavailable stubs implement the seam;
  provider contracts fully overwrite before reading; exact-head WGPU, CUDA,
  ROCm, and Metal CI passes; Coeus uses the seam for shared-storage copies
  after the provider merge; ADR and changelog are synchronized.
- Risk/change class: `[arch]`/`[minor]` additive provider seam with a strict
  memory-initialization contract; ADR 0037 owns the decision.
- Status: complete. Hephaestus PR #136 and Coeus PR #235 merged after the
  provider and consumer cutovers. Provider run `30343728210` passed CUDA job
  `90224950226`, run `30343728174` passed WGPU job `90224950173`, run
  `30343728133` passed ROCm job `90224950310`, and run `30343728161` passed
  Metal job `90224950041`. Coeus final docs-head run `30346488092` passed
  CUDA `90233799719`, WGPU `90233799768`, ROCm `90233799650`, and Metal
  `90233799737`; required-device ROCm `90233800152` skipped because no hosted
  AMD runner was dispatched. NVIDIA `90224951166` and AMD `90224950902` also
  skipped because no physical-device runner was dispatched.

## HEPH-DEVICE-LOCAL-COW-1 [arch] [patch] — done

- Owner: Codex on `codex/device-local-cow-copy`; scope: the shared
  `ComputeDevice` copy contract and the Coeus Hephaestus storage uniqueness
  consumer.
- Outcome: detach shared accelerator storage with a provider-native
  device-to-device copy, retaining the source memory tier and eliminating the
  full-size temporary host allocation.
- Non-goals: asynchronous storage mutation, new provider kernels, CPU
  fallback, and runtime speed or allocation claims without controlled
  measurements.
- Acceptance: WGPU, CUDA, ROCm, and Metal implement the shared copy contract;
  unavailable configurations return typed errors; Coeus `make_unique` uses one
  device allocation and device-local copy; focused provider and consumer gates
  pass; ADR and changelog are synchronized.
- Risk/change class: `[arch]`/`[patch]` additive backend seam with a memory
  lifetime contract; ADR 0036 owns the decision.
- Status: complete. Hephaestus PR #134 merged at `24439bf`. The exact-head
  provider lanes passed CUDA job `90198606004`, ROCm job `90198605585`, WGPU
  job `90198605775`, and macOS Metal job `90198605623`. AMD and NVIDIA
  required-device jobs were skipped because hosted hardware runners were not
  available; no physical-device execution claim is made. The recurring
  `recurseml/analysis` service error is not repository-owned verification.

## HEPH-DENSE-VECTOR-ELEMENTWISE-1 [minor] — done

- Owner: Codex on `codex/dense-vector-elementwise-parity`; scope: the shared
  `DenseVectorOps` elementwise contract plus WGPU, CUDA, ROCm, and Metal
  implementations and value-semantic tests.
- Outcome: expose the remaining Leto dense-slice arithmetic operations through
  one caller-owned-output seam without per-operation result allocation or
  provider-local algorithm copies.
- Non-goals: non-f32 vector scalars, reductions beyond the existing prepared
  dot/L2 contract, Coeus routing, and new kernels where existing provider
  elementwise paths already satisfy the operation.
- Acceptance: all four providers implement `add_into`, `subtract_into`,
  `multiply_into`, and `divide_into`; tests cover CPU values, empty inputs,
  output-buffer reuse, workgroup tails, and typed length rejection; exact-head
  WGPU, CUDA, ROCm, and Metal CI passes.
- Risk/change class: `[minor]` additive consumer-facing vector capability.
- Status: complete. Merged in Hephaestus PR #132 at `7c481b2`. The exact-head
  provider matrix passed CUDA job `90190923765`, ROCm job `90190923670`, WGPU
  job `90190923733`, and macOS Metal job `90190923914`; required-device CUDA
  and ROCm jobs were skipped because hosted hardware runners were unavailable.
  The external `recurseml/analysis` check returned its recurring analysis-
  service error and is not repository-owned verification.

## HEPH-DENSE-VECTOR-OPS-METAL-1 [arch] [minor] — done

- Owner: Codex; scope: the Metal provider implementation and its
  value-semantic contract tests.
- Outcome: expose the complete dense-vector operation seam through the
  native-Metal-selected dispatch substrate without cloned consumer algorithms
  or host fallback.
- Non-goals: non-f32 vector scalars, Coeus routing, and new Metal-specific
  kernels where the canonical WGPU kernels already execute on Metal.
- Acceptance: Metal implements copy, scale, AXPY, XPAY, subtraction, prepared
  dot, and prepared L2 norm; prepared handles retain no additional buffer
  clones; contracts compare CPU values, cover empty vectors and workgroup
  tails, and exercise prepared reuse; exact-head Metal and sibling backend CI
  passes.
- Risk/change class: `[arch]`/`[minor]` provider seam completion; ADR 0034
  owns the delegation and prepared-resource decision.
- Status: merged in PR #131 at `36021feac2dd05e8a1b4e3621804a05456f8ff39`.
- Evidence: WGPU, CUDA, ROCm, and macOS Metal repository-owned checks passed;
  NVIDIA and AMD hardware jobs were skipped by workflow inputs.

## HEPH-DENSE-VECTOR-OPS-PARITY-1 [arch] [minor] — done

- Owner: Codex; scope: the shared `DenseVectorOps` GAT contract plus CUDA and
  ROCm provider implementations.
- Outcome: expose one value-semantic dense-vector operation seam across WGPU,
  CUDA, and ROCm without CPU fallback or cloned consumer algorithms.
- Non-goals: Metal implementation, non-f32 vector scalars, and Coeus routing.
- Acceptance: CUDA and ROCm implement copy, scale, AXPY, XPAY, subtraction,
  prepared dot, and prepared L2 norm; prepared plans borrow fixed allocations
  while owning only cheap device handles; provider contracts compare values
  with CPU formulas and exercise workgroup tails; exact-head WGPU, CUDA, ROCm,
  and Metal CI passes.
- Risk/change class: `[arch]`/`[minor]` public GAT lifetime change; ADR 0033
  owns the ownership and memory-efficiency decision.
- Status: merged in PR #130 at `cf28aa594f1348f39c6510fc2bb30edf943a3f37`.
- Evidence: WGPU, CUDA, ROCm, and Metal repository-owned PR checks passed;
  NVIDIA and AMD hardware jobs were skipped by workflow input.

## HEPH-LGAMMA-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex; scope: shared `LgammaOp` marker and WGPU, CUDA, ROCm, and Metal
  exports consumed by Coeus.
- Outcome: expose Leto's `ln|Gamma(x)|` operation through one provider-owned
  accelerator vocabulary without a host fallback.
- Non-goals: digamma gradients, f64/reduced/vector result contracts, and
  unrelated operation families.
- Acceptance: native CUDA/HIP expressions and a WGSL Lanczos/reflection
  expression are exported by all four providers; Coeus routes f32 `lgamma`
  through every backend and compares positive, reflected, and pole inputs with
  the Leto CPU oracle; exact-head provider CI passes.
- Risk/change class: `[minor]` additive operation vocabulary; ADR 0031 owns the
  mathematical and expression-contract decision.
- Status: complete. Hephaestus provider implementation and exact-head provider
  verification are complete; Coeus routing and Leto differential verification
  merged in Coeus PR #231.
- Evidence: Hephaestus PR #118 provider jobs passed WGPU `90086428952`, CUDA
  `90086430178`, ROCm `90086430143`, and Metal `90086428160`. Coeus PR #231
  merged at `971fab9614b97bd708a716d01684da58fd1331ba`; its consumer jobs
  passed WGPU `90088836682`, CUDA `90088836688`, ROCm `90088836731`, and
  Metal `90088836675`. Required-device ROCm job `90088837591` was skipped;
  no physical-device execution claim is made.

## HEPH-GELU-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex; scope: shared exact `GeluOp` and `GeluGradOp` markers and WGPU,
  CUDA, ROCm, and Metal exports consumed by Coeus.
- Outcome: expose the Leto CPU exact GELU forward and gradient vocabulary to
  every accelerator provider without a host fallback.
- Non-goals: tanh-approximated GELU, parameterized activations, `lgamma`,
  tail-stable `erfc`, f64/vector result contracts, and unrelated operation
  families.
- Acceptance: markers implement `0.5*x*(1+erf(x/sqrt(2)))` and its analytic
  derivative in WGSL, CUDA C++, and HIP C++; all four provider roots export
  both markers; Coeus ROCm and Metal route both operations and compare them
  with the Leto CPU oracle; exact-head WGPU, CUDA, ROCm, and Metal CI passes.
- Risk/change class: `[minor]` additive operation vocabulary; ADR 0030 owns the
  mathematical and expression-contract decision.
- Status: complete. Hephaestus provider implementation and exact-head provider
  verification are complete; Coeus routing and Leto differential verification
  merged in Coeus PR #230 at `e26ba668`.
- Evidence: merged Hephaestus PR #123 at `23f9662` passed WGPU job
  `90115184352`, CUDA job `90115253352`, ROCm job `90115183816`, and Metal job
  `90115184178`. AMD and NVIDIA hardware jobs were skipped; no physical-device
  execution claim is made. Coeus consumer jobs passed WGPU `90061390522`,
  CUDA `90061390565`, ROCm `90061390546`, and Metal `90061390499`.

## HEPH-ACTIVATION-TAIL-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex; scope: shared `MishOp`, `MishGradOp`, `EluOp`, and `EluGradOp`
  markers and WGPU, CUDA, ROCm, and Metal exports consumed by Coeus.
- Outcome: expose the remaining parameter-free activation expressions in one
  provider-owned vocabulary without a host fallback.
- Non-goals: parameterized activations, alternate stabilized softplus forms,
  f64/reduced/vector result contracts, and unrelated operation families.
- Acceptance: each marker has WGSL, CUDA C++, and HIP C++ expressions; all
  four provider roots export the markers; provider contracts compare forward
  and gradient results with the f32 CPU formulas; Coeus routes the operations
  through every backend and compares them with the Leto CPU oracle; exact-head
  WGPU, CUDA, ROCm, and Metal CI passes.
- Risk/change class: `[minor]` additive operation vocabulary; ADR 0032 owns
  the mathematical and expression-contract decision.
- Status: complete. Hephaestus provider implementation, local contracts, and
  exact-head provider verification are complete; Coeus routing and Leto
  differential verification merged in Coeus PR #237 at `7fef4a2a`.
- Evidence: merged Hephaestus PR #123 at `23f9662`; WGPU job `90115184352`
  (pass), CUDA job `90115253352` (pass), ROCm job `90115183816` (pass), and
  Metal job `90115184178` (pass). AMD and NVIDIA hardware jobs were skipped;
  no physical-device execution claim is made. Exact Coeus consumer head
  `2f04be65` passed WGPU `90262230232`, CUDA `90262230288`, ROCm
  `90262230238`, and Metal `90262230226`.

## HEPH-ERROR-FUNCTION-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex; scope: shared `ErfOp` and `ErfcOp` markers and WGPU, CUDA, ROCm,
  and Metal exports consumed by Coeus.
- Outcome: expose one provider-owned expression vocabulary for the Leto CPU
  `erf` and `erfc` operations.
- Non-goals: `lgamma`, tail-stable `erfc`, parameterized activations, f64 or
  vector result contracts, and unrelated unary or higher-rank operation
  families.
- Acceptance: each marker has the existing WGPU approximation or native CUDA
  C++/HIP C++ expression; all four backend application modules export both
  markers; core expression tests pass; exact-head WGPU, CUDA, ROCm, and Metal
  CI passes; Coeus routes the f32 operations through ROCm and Metal and
  compares them with the Leto CPU oracle.
- Risk/change class: `[minor]` additive operation vocabulary; ADR 0029 owns the
  expression and numerical-contract decision.
- Status: complete. Hephaestus implementation and exact-head provider evidence
  are complete, and Coeus routes the f32 operations through ROCm and Metal.
- Evidence: provider docs head `df8a896` passed WGPU job `90028947591`, CUDA
  job `90028946846`, ROCm job `90028946770`, and Metal job `90028947450`.
  Coeus PR #228 merged at `aca9a5a8`; its final docs head `08614299` passed
  run `30283857017` with CUDA job `90036655765`, ROCm job `90036655656`, Metal
  job `90036655618`, and WGPU job `90036655846`. AMD/NVIDIA hardware jobs
  skipped because no registered device runner was selected.

## HEPH-UNARY-MATH-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex on `codex/hephaestus-unary-math-parity`; scope: shared
  unparameterized f32 unary math expression markers and WGPU, CUDA, ROCm, and
  Metal exports consumed by Coeus.
- Outcome: expose one dialect-specific marker for tangent, inverse and
  hyperbolic functions, logarithm/exponential bases, sign, and rounding
  operations so the four providers share one monomorphized operation
  vocabulary.
- Non-goals: `erf`, `erfc`, `lgamma`, parameterized activations, f64 or vector
  result contracts, and unrelated unary or higher-rank operation families.
- Acceptance: each marker has WGSL, CUDA C++, and HIP C++ expressions; all four
  backend application modules export the same markers; core expression tests
  pin direct, composed, sign, and rounding forms; Coeus routes the f32
  operations through ROCm and Metal and compares them with the Leto CPU
  oracle; exact-head WGPU, CUDA, ROCm, and Metal CI passes.
- Risk/change class: `[minor]` additive operation vocabulary; ADR 0028 owns the
  f32 expression and residual-capability decision.
- Status: complete. Hephaestus implementation is complete at `b088a2f`; WGPU job
  `89997918070`, CUDA job `89997916944`, ROCm job `89997920644`, and Metal job
  `89997917574` all passed. Required AMD and NVIDIA hardware jobs were skipped;
  Coeus PR #226 merged at `383ac51b`; exact consumer head `7c9a1ab2` passed
  WGPU `90015947922`, CUDA `90015947631`, ROCm `90015947658`, and Metal
  `90015947762`.

## HEPH-COMPARISON-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex; scope: typed comparison expressions and WGPU, CUDA, ROCm, and
  Metal binary/strided provider exports used by Coeus.
- Outcome: provide native equality, inequality, ordering, and inclusive-order
  comparisons for f32, i32, and u32 with scalar-correct WGSL, CUDA C++, and
  HIP C++ mask expressions.
- Non-goals: f64/vector comparison result contracts, parameterized activations,
  exact-erf GELU, and unrelated higher-rank or matrix operation families.
- Acceptance: Hephaestus core expression tests pass; Coeus ROCm and Metal
  providers route all six comparisons through typed Hephaestus kernels; Leto
  differential tests cover f32 broadcast plus i32/u32 values; exact-head
  WGPU, CUDA, ROCm, and Metal CI passes.
- Risk/change class: `[minor]` additive typed operation vocabulary.
- Status: merged in Hephaestus PR #111 at `14f8972`; final implementation head
  `50713d8` passed WGPU job `89867844717`, CUDA job `89867844583`, ROCm job
  `89867844846`, and Metal job `89867844633`. Coeus PR #224 merged at
  `84b5bcc`; exact-head run `30268824209` passed CUDA job `89986119939`, WGPU
  job `89986119972`, Metal job `89986119988`, and ROCm job `89986120026`.
  Required-device AMD and NVIDIA lanes were skipped because hosted hardware
  runners were unavailable; no physical-device execution claim is made.
- Decision: ADR 0027 selects `TypedBinaryExpr<L, T>` over f32-only markers or
  per-backend comparison kernels.

## HEPH-ACTIVATION-EXPRESSION-PARITY-1 [minor] — done

- Owner: Codex; scope: `hephaestus-core` activation expression vocabulary and
  WGPU, CUDA, ROCm, and Metal exports used by Coeus native providers.
- Outcome: provide one dialect-specific ZST marker for each common activation
  forward and gradient operation, preserving monomorphized dispatch across
  WGSL, CUDA C++, and HIP C++.
- Non-goals: parameterized activations, exact-erf GELU, comparisons, and
  unrelated higher-rank or matrix operation families.
- Acceptance: ReLU, sigmoid, tanh, tanh-GELU, SiLU, and softplus forward and
  gradient markers compile for all three kernel dialects; Hephaestus core
  expression tests pass; Coeus ROCm and Metal providers consume the markers
  without host fallback.
- Risk/change class: `[minor]` additive operation vocabulary.
- Status: complete. Merged in Hephaestus PR #110; the exact code head passed WGPU job
  `89857349160`, CUDA job `89857348956`, ROCm job `89857348904`, and Metal job
  `89857349033`. Coeus PR #223 merged at `4b807ddd`; exact consumer head
  `3c53a8ee` passed WGPU `89860389892`, CUDA `89860389857`, ROCm
  `89860389885`, and Metal `89860389899`.

## HEPH-SCAN-SUFFIX-PARITY-1 [minor] — done

- Owner: Codex on `codex/hephaestus-suffix-scan-reconcile`; scope: rank-2
  `suffix_sum`/`suffix_sum_into` exports and provider contracts in WGPU, CUDA,
  ROCm, and Metal. Coeus consumer routing is tracked in
  `ATLAS-HEPHAESTUS-SCAN-001` in the Coeus repository.
- Non-goals: dynamic-rank scan expansion and unrelated Leto operation families.
- Acceptance: all four roots expose the same suffix-sum API, each provider
  delegates to the shared reverse cumulative-sum kernel, and each contract
  compares allocated and caller-owned outputs with the Leto CPU oracle.
- Risk/change class: `[minor]` additive provider surface.
- Status: complete. PR #107 merged the shared reverse cumulative-sum APIs and
  Leto differential contracts; PR #122 merged the terminal provider evidence.
  Exact closeout head `45079f1d` passed WGPU `90109098364`, CUDA
  `90109098361`, ROCm `90109098296`, and macOS Metal `90109098891`.
  Hardware-only NVIDIA and AMD jobs skipped because no device runner was
  selected.

## HEPH-BACKEND-CI-FEATURE-MATRIX-1 [patch] — done

- Owner: Codex; scope: standalone WGPU provider CI, CUDA decomposition feature
  wiring, and synchronized verification artifacts.
- Non-goals: new numerical kernels, provider-specific fallbacks, or changes to
  the established common operation surface.
- Acceptance: WGPU's default and minimal feature surfaces run the committed
  warning-denied, Nextest, doctest, and rustdoc gates in CI; CUDA's
  `decomposition` feature necessarily enables its `cuda` substrate; the four
  backend operation matrix remains complete; provider CI passes on the exact
  head.
- Verification: local format, metadata, feature-graph, and focused package
  gates where the checkout-local dependency graph permits; WGPU, CUDA, ROCm,
  and Metal workflow results recorded on the final head.
- Evidence: final head `6d9e96f` passed WGPU run `30187568513` / job
  `89754890646` (6m10s), CUDA run `30187568521` / job `89754910876`
  (7m11s), ROCm run `30187568515` / job `89754890592` (5m40s), and macOS
  Metal run `30187568510` / job `89754890620` (6m57s). NVIDIA hardware job
  `89754911060` and AMD hardware job `89754890775` were skipped because
  hosted hardware labels were unavailable; no hardware execution claim is
  made.
- Claimed files: `crates/hephaestus-cuda/Cargo.toml`,
  `.github/workflows/wgpu.yml`, `README.md`, `CHANGELOG.md`, `backlog.md`,
  `checklist.md`, and `gap_audit.md`. Execution owner: Codex on
  `codex/hephaestus-backend-parity-next-9`; completed: 2026-07-26.

## HEPH-BACKEND-PARITY-MATRIX-1 [patch] — done

- Owner: Codex; scope: audit the four backend crate-root operation surfaces
  after the Metal output-buffer reduction closure. Common capability is
  defined by lowercase operation exports from the application modules;
  provider-specific types and CUDA runtime-shaped consumer helpers are
  excluded from the common baseline.
- Acceptance: WGPU, CUDA, ROCm, and Metal share every operation in the common
  baseline; the scan reports no missing WGPU baseline operation; provider CI
  passes on the exact documented head.
- Verification: fresh `origin/master` at `0350838` exposes 93 common lowercase
  operations across all four roots. CUDA additionally exposes the intentional
  dynamic-rank helpers `binary_elementwise_strided_dyn_into` and
  `unary_elementwise_strided_dyn_into`; no WGPU baseline operation is missing
  from CUDA, ROCm, or Metal. Final docs-head `2b9f162` passed CUDA run
  `30183321092` / job `89743580678` (7m25s), ROCm run `30183321132` / job
  `89743580809` (5m08s), and macOS Metal run `30183321100` / job
  `89743580707` (5m30s). NVIDIA and AMD hardware jobs were skipped because
  hosted hardware labels were unavailable.
- Claimed files: `gap_audit.md`, `checklist.md`, and this item. Execution
  owner: Codex on `codex/hephaestus-backend-parity-next-8`; completed:
  2026-07-26.

## HEPH-BACKEND-PARITY-METAL-REDUCE-INTO-1 [minor] — done

- Owner: Codex; scope: export Metal's existing `reduce_axis_into` operation
  from the crate root and add a direct value-semantic rank-2 axis-reduction
  contract matching WGPU, CUDA, and ROCm. New reduction kernels, fallback
  paths, and unrelated reduction APIs are non-goals.
- Acceptance: the four backend roots export `reduce_axis_into`; Metal's
  output-buffer path produces the CPU-reference axis result; Metal provider
  CI passes.
- Verification: implementation head `a8ad020` passed CUDA feature and
  adapterless contracts (run `30183081825`, job `89742932396`, 7m31s), ROCm
  feature and adapterless contracts (run `30183081834`, job `89742932370`,
  5m53s), and macOS Metal contracts (run `30183081848`, job `89742932459`,
  4m57s). NVIDIA hardware (job `89742932642`) and AMD hardware (job
  `89742932594`) were skipped because hosted hardware labels were unavailable;
  no hardware execution claim is made. RecurseML reported a generic analysis
  service error without a source diagnostic.
- Claimed files: Metal root export and contract test, README, CHANGELOG,
  `docs/adr/0025-metal-reduce-into-parity.md`, `checklist.md`, and this item.
  Execution owner: Codex on `codex/hephaestus-backend-parity-next-7`;
  completed: 2026-07-26.

## HEPH-BACKEND-PARITY-BLOCKED-PIVOTED-1 [minor] — done

- Owner: Codex; scope: expose blocked complete-pivot LU and blocked
  column-pivoted QR with the same dense C-contiguous contract on WGPU, CUDA,
  ROCm, and Metal. New pivoting algorithms, fallback paths, and unrelated
  decomposition families are non-goals.
- Acceptance: all four backend roots export `full_piv_lu_blocked` and
  `col_piv_qr_blocked`; dense inputs produce the same factor values,
  permutations, ranks, and solve contracts as the corresponding ordinary
  entry points; non-dense inputs are rejected with the typed dense-layout
  error; WGPU, CUDA, ROCm, and macOS Metal provider CI passes.
- Verification: implementation head `5314522` passed CUDA feature and
  adapterless contracts (run `30182486511`, job `89741393411`, 7m35s), ROCm
  feature and adapterless contracts (run `30182486506`, job `89741368781`,
  5m53s), and macOS Metal contracts (run `30182486494`, job `89741368870`,
  6m14s). NVIDIA hardware (job `89741391064`) and AMD hardware (job
  `89741368976`) were skipped because hosted hardware labels were unavailable;
  no hardware execution claim is made.
- Claimed files: backend decomposition modules and exports, provider contract
  tests, README, CHANGELOG, `docs/adr/0024-blocked-pivoted-parity.md`,
  `checklist.md`, and this item. Execution owner: Codex on
  `codex/hephaestus-backend-parity-next-6`; completed: 2026-07-26.

## HEPH-BACKEND-PARITY-METAL-EXP-NEG-1 [minor] — done

- Owner: Codex; scope: expose the existing fused `ExpNegOp` (`exp(-x)`) marker
  through Metal and add its value-semantic contract. New unary kernels,
  alternate exponential semantics, and unrelated operator families are
  non-goals.
- Acceptance: Metal exports `ExpNegOp` alongside the other unary markers; the
  existing Metal generic unary path dispatches it through the native
  Metal-selected WGPU device; the contract compares the fused output with an
  independent CPU `(-x).exp()` oracle and the `NegOp` then `ExpOp` composition;
  and the existing CUDA, ROCm, and macOS Metal CI lanes pass.
- Verification: code head `f8abf64` passed CUDA feature and adapterless
  contracts (run `30181256267`, job `89738055998`, 8m07s), ROCm feature and
  adapterless contracts (run `30181256264`, job `89738055911`, 5m50s), and
  macOS Metal contracts (run `30181256288`, job `89738056078`, 4m59s). NVIDIA
  hardware (job `89738056588`) and AMD hardware (job `89738056328`) were
  skipped because hosted hardware labels were unavailable; no hardware
  execution claim is made.
- Claimed files: Metal elementwise exports and contract test, README,
  CHANGELOG, `docs/adr/0023-metal-exp-neg-parity.md`, `checklist.md`, and this
  item. Execution owner: Codex on
  `codex/hephaestus-backend-parity-next-5`; completed: 2026-07-25.

## HEPH-BACKEND-PARITY-METAL-AUTHORED-1 [minor] — done

- Owner: Codex; scope: expose Metal-owned authored-kernel command streams and
  storage-kernel dispatch matching the existing WGPU, CUDA, and ROCm provider
  seams. New kernel algorithms, a second native `metal-rs` implementation, and
  unrelated operator families are non-goals.
- Acceptance: Metal implements `KernelDevice` and
  `GroupedKernelDevice` with ordered dispatch, copy, prefix-copy, zero-fill,
  and grouped sequence behavior; Metal exports multi-storage, unary, and
  binary storage-kernel wrappers with typed binding validation; value-semantic
  Metal contracts cover dispatch, copy/fill, grouped output, storage output,
  and invalid lengths; and the existing Metal CI lane runs feature, lint,
  Nextest, doctest, and rustdoc gates.
- Verification: implementation head `d200879` passed CUDA feature and
  adapterless contracts (run `30180682356`, job `89736582959`, 7m39s), ROCm
  feature and adapterless contracts (run `30180682371`, job `89736559954`,
  5m52s), and macOS Metal contracts (run `30180682365`, job `89736559801`,
  5m17s). NVIDIA hardware (job `89736583231`) and AMD hardware (job
  `89736560290`) were skipped because hosted hardware labels were unavailable;
  no hardware execution claim is made.
- Claimed files: Metal authored-kernel and storage-kernel modules/exports,
  Metal contract tests, README, CHANGELOG, `docs/adr/0022-metal-authored-kernel-parity.md`,
  `checklist.md`, and this item. Execution owner: Codex on
  `codex/hephaestus-backend-parity-next-4`; completed: 2026-07-25.

## HEPH-BACKEND-PARITY-METAL-FLUENT-LINALG-1 [minor] — done

- Owner: Codex; scope: expose Metal-owned fluent dense-matrix traits matching
  the existing WGPU, CUDA, and ROCm method families for operand conversion,
  products, norms, decomposition, solves, matrix properties, and matrix
  functions. New Metal kernels, shared-algorithm rewrites, and dynamic-rank
  helpers are non-goals.
- Acceptance: Metal exports `AsGpuMatrixOperand`, `MatrixProduct`,
  `MatrixNorm`, `MatrixDecompose`, `MatrixSolve`, `MatrixProperties`, and
  `MatrixFunction` with Metal device and buffer contracts; every method
  delegates to an existing Metal-selected operation; decomposition handles
  retain the existing shared WGPU-backed result types; value-semantic Metal
  tests cover representative product, norm, property, function, solve, and
  decomposition calls; and the Metal CI lane passes its feature, lint,
  Nextest, doctest, and rustdoc gates.
- Claimed files: Metal linalg traits and exports, Metal contract tests,
  README, CHANGELOG, `docs/adr/0021-metal-fluent-linalg-parity.md`,
  `checklist.md`, and this item. Execution owner: Codex on
  `codex/hephaestus-backend-parity-next-3`; last update: 2026-07-25.
- Hosted code-head evidence at `8491592`: CUDA run `30179437377` / job
  `89733458943` passed in 7m24s, ROCm run `30179437392` / job `89733459927`
  passed in 6m01s, and Metal run `30179437384` / job `89733463248` passed in
  5m09s. The AMD and NVIDIA required-device jobs were skipped because hosted
  GPU labels were unavailable; provider lanes ran their real feature,
  warning-denied Clippy, Nextest, doctest, and rustdoc gates. Earlier Metal
  failures were fixed at their causes: the private module export at
  `5b83626`, then the default-feature unused import at `8491592`.

## HEPH-BACKEND-PARITY-METAL-RANDOM-1 [minor] — done

- Owner: Codex; scope: expose deterministic seeded uniform and normal
  initializers through Metal, matching the existing WGPU, CUDA, and ROCm
  application contract. New random algorithms, device-native PRNG kernels,
  and fluent matrix traits are non-goals.
- Acceptance: Metal exports `uniform_with_seed` and `normal_with_seed` for
  the same real scalar and const-generic rank contract; the implementation
  delegates deterministic value generation to the existing WGPU application
  path on Metal's native device and uploads the result into `MetalBuffer<T>`;
  contracts verify deterministic repeated seeds, uniform bounds, nonzero
  normal output, and the existing required-device behavior; the Metal CI lane
  executes the feature, lint, nextest, doctest, and rustdoc gates.
- Claimed files: Metal random module/export/tests, README, CHANGELOG,
  `docs/adr/0020-metal-random-parity.md`, `checklist.md`, and this item.
  Execution owner: Codex on `codex/hephaestus-backend-parity-next-2`; last
  update: 2026-07-25.
- Hosted code-head evidence at `96abaac`: CUDA run `30178303257` / job
  `89730624710` passed in 7m12s, ROCm run `30178303267` / job `89730603655`
  passed in 5m57s, and Metal run `30178303260` / job `89730601126` passed in
  6m15s. The AMD and NVIDIA required-device jobs were skipped because hosted
  GPU labels were unavailable; the provider lanes ran their real feature,
  warning-denied Clippy, Nextest, doctest, and rustdoc gates.

## HEPH-BACKEND-PARITY-SCAN-PRODUCT-1 [minor] — done

- Owner: Codex; scope: expose the existing rank-2 cumulative-product scan
  contract through WGPU, CUDA, ROCm, and Metal convenience APIs, with
  value-semantic contracts and synchronized documentation. New scan kernels
  and unrelated operator families are non-goals.
- Acceptance: all four backends expose forward `cumprod`/`cumprod_into` and
  reverse `suffix_prod`/`suffix_prod_into` with the same rank-2 strided
  input/output, axis, and width contract; each provider delegates to its
  generic `CumProdOp` scan path; tests compare allocated and caller-owned
  outputs to the Leto CPU oracle over both directions, both axes,
  non-contiguous storage, empty inputs, and invalid layouts; the provider CI
  lanes execute the contracts and documentation records the public parity
  surface.
- Claimed files: backend scan modules and exports, backend scan contract
  tests, CHANGELOG, `docs/adr/0019-scan-product-parity.md`,
  `checklist.md`, and this item. Execution owner: Codex on
  `codex/hephaestus-leto-coeus-parity-next`; last update: 2026-07-26.
- Hosted evidence is closed by the exact merged PR #132 head `7c481b2`:
  CUDA job `90190923765`, ROCm job `90190923670`, WGPU job `90190923733`, and
  macOS Metal job `90190923914` passed their complete provider suites. The
  required-device CUDA and ROCm jobs were skipped because hosted hardware
  runners were unavailable; no hardware execution claim is made.

## HEPH-BACKEND-PARITY-PREPARED-MAP-REDUCTION-1 [minor] — done

- Owner: Codex; scope: prepared dot and L2-norm map-reduction plans across
  WGPU, CUDA, ROCm, and Metal, including fixed product scratch, reduction
  trees, scalar outputs, repeated dispatch, non-contiguous layouts, tests,
  CI, and synchronized documentation. Prepared L1/max/trace plans, new
  numerical algorithms, and Python API changes are non-goals.
- Acceptance: CUDA and ROCm expose `PreparedDot`, `prepare_dot`,
  `PreparedL2Norm`, and `prepare_norm_l2` over fixed strided buffers; plans
  retain the mapped product scratch, reduction tree, and output allocations;
  repeated dispatch observes input updates without changing output identity;
  empty and non-contiguous layouts preserve CPU-reference values; invalid
  shapes/layouts reject before launch; Metal exposes the same public surface
  through the native WGPU-Metal path; and CUDA/ROCm/Metal CI runs the focused
  feature, lint, nextest, doctest, and rustdoc gates.
- Claimed files: CUDA/ROCm prepared reduction-plan extraction and prepared
  map-reduction modules/exports/tests, Metal delegation/exports/tests, README,
  changelog, ADR 0018, checklist, backlog, and existing backend CI workflows.
  Last update: 2026-07-25. Hosted feature lanes passed at code head `2af0c72`:
  CUDA run `30176531577` / job `89726109677` (7m44s), ROCm run
  `30176531572` / job `89726109624` (5m37s), and Metal run `30176531566` /
  job `89726137236` (4m57s). Hardware lanes were skipped because hosted GPU
  labels were unavailable; required-device enforcement remains enabled for
  self-hosted runners. Local Windows package compilation remains blocked before
  source compilation by the locked `cutile-rs` refresh and the sibling
  Leto/Eunomia `Quantity<T>::in_unit` / `FloatElement` mismatch.

## HEPH-BACKEND-PARITY-PREPARED-AXIS-1 [minor] — done

- Owner: Codex; scope: prepared rank-2 axis reduction plans for WGPU, CUDA,
  ROCm, and Metal, including fixed input/output layouts, retained native
  pipeline and metadata resources, repeated dispatch, batch submission,
  value-semantic contracts, existing backend CI lanes, and synchronized
  documentation. Prepared sparse operations and map-reduction plans are
  non-goals for this increment.
- Acceptance: CUDA and ROCm expose the same prepared rank-2 axis capability as
  WGPU/Metal for sum, min, max, and mean; preparation reuses the shared axis
  planner and validates axis, shape, layout, width, and alias contracts;
  dispatch launches only retained device resources without host materialization;
  repeated dispatch and batch submission preserve value semantics across both
  axes and non-contiguous layouts; sum preserves its identity on empty
  reduced axes while min/max/mean reject undefined empty axes; and
  CUDA/ROCm/Metal CI runs the focused contracts.
- Claimed files: CUDA/ROCm prepared-axis modules and exports/tests, Metal
  delegation and exports/tests, WGPU empty-axis binding fix and regression test,
  affected CI/docs, and this item. Last update: 2026-07-25. Hosted feature
  lanes passed at code head `7728026`: CUDA run `30173211020` / job
  `89717642298`, ROCm run `30173211026` / job `89717642411`, and Metal run
  `30173211021` / job `89717642467`. Hardware lanes were skipped because
  hosted GPU labels were unavailable; required-device enforcement remains
  enabled for self-hosted runners. Local Windows package compilation remains
  blocked before source compilation by the locked `cutile-rs` refresh and the
  sibling Leto/Eunomia `Quantity<T>::in_unit` / `FloatElement` mismatch.

## HEPH-BACKEND-PARITY-PREPARED-SPARSE-1 [minor] — done

- Owner: Codex; scope: prepared CSR SpMV, SpMM, multi-RHS SpMV, and one-submit
  batching across WGPU, CUDA, ROCm, and Metal, including backend-owned CSR
  storage wrappers, fixed device operands/output buffers, value-semantic
  contracts, CI, and synchronized documentation. New sparse algorithms,
  map-reduction plans, and Python API changes are non-goals.
- Acceptance: CUDA and ROCm expose prepared `spmv`, `spmm`, and `spmv_many`
  plans that retain validated CSR metadata, native compiled kernels, and fixed
  dense operands/output storage; Metal exposes the same public sparse contract
  by delegating through its native WGPU-Metal device; all four backends expose
  equivalent prepared operation and batch names; repeated and mixed SpMV/SpMM
  batches preserve CPU-reference values for dense and non-contiguous RHS
  layouts; invalid shapes, layouts, and output lengths reject before launch;
  and CUDA/ROCm/Metal CI runs focused feature, lint, nextest, doctest, and
  rustdoc gates.
- Claimed files: CUDA/ROCm sparse prepared modules and exports/tests, Metal
  sparse wrapper, CSR buffer accessors, affected README/changelog, ADR 0017,
  checklist, backlog, and existing backend CI workflows. Last update:
  2026-07-25. Hosted feature lanes passed at code head `27bf875`: CUDA run
  `30175194827` / job `89722726883` (7m17s), ROCm run `30175194823` / job
  `89722726994` (5m44s), and Metal run `30175194820` / job `89722726870`.
  NVIDIA and AMD hardware lanes were skipped because hosted GPU labels were
  unavailable; required-device enforcement remains enabled for self-hosted
  runners. Local Windows package compilation remains blocked before source
  compilation by the locked `cutile-rs` refresh and the sibling
  Leto/Eunomia `Quantity<T>::in_unit` / `FloatElement` mismatch.

## HEPH-METAL-CI-1 [patch] — done

- Owner: Codex; scope: existing `hephaestus-metal` WGPU-Metal backend
  documentation, required-device contract behavior, macOS Metal feature/build/
  test/doc CI, and synchronized checklist/changelog state. A second native
  `metal-rs` implementation and new Metal operator families are non-goals for
  this increment.
- Acceptance: the public backend inventory names `hephaestus-metal`; its
  ownership is documented as WGPU configured for native Metal; hardware CI
  sets `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` so unavailable hardware fails rather
  than skips; macOS CI checks the default and minimal feature surfaces, runs
  warning-denied Clippy, required-device Nextest, doctest, and rustdoc; and the
  Metal contract tests retain value-semantic device/CPU checks.
- Claimed files: `crates/hephaestus-metal/tests/contract.rs`,
  `.github/workflows/metal.yml`, `README.md`, `CHANGELOG.md`, `checklist.md`,
  and this item. Last update: 2026-07-25. Hosted macOS Metal job
  `89630859643` passed at PR head `9292c20`, including required-device
  contracts, feature builds, warning-denied Clippy, doctest, and rustdoc.
  The ROCm container job `89630859765` also passed after the shared
  `mnemosyne-core` source patch. Local Windows package compilation remains
  blocked by the sibling Leto/Eunomia `Quantity<T>::in_unit` /
  `FloatElement` mismatch; hosted checkout-graph CI is the accepted gate.

## HEPH-BACKEND-PARITY-PREPARED-REDUCE-1 [minor] — done

- Owner: Codex; scope: prepared scalar reduction plans for WGPU, CUDA, ROCm,
  and Metal, including reusable device-resident scratch/output storage,
  repeated dispatch, batch submission, value-semantic contracts, existing
  backend CI lanes, and synchronized documentation. Prepared axis reductions,
  prepared sparse operations, and map-reduction plans are non-goals for this
  increment.
- Acceptance: CUDA and ROCm expose the same prepared scalar reduction
  capability as WGPU/Metal for sum, min, and max; preparation validates the
  block width and allocates the complete reduction tree; dispatch launches
  only the prepared device kernels without host materialization; repeated
  dispatch reuses the output allocation; empty, singleton, multi-pass, and
  invalid-width contracts are value-tested; and CUDA/ROCm/Metal CI runs the
  focused contracts.
- Claimed files: CUDA/ROCm prepared-reduction modules and exports/tests,
  Metal delegation and exports/tests, affected CI/docs, and this item. Last
  update: 2026-07-25. Hosted feature lanes passed at code head `4279244`:
  CUDA run `30171447372` / job `89713176239`, ROCm run `30171447427` /
  job `89713176460`, and Metal run `30171447381` / job `89713176351`.
  Hardware lanes were skipped because hosted GPU labels were unavailable;
  required-device enforcement remains enabled for self-hosted GPU runners.
  Local Windows package compilation remains blocked before source compilation
  by the locked `cutile-rs` git dependency refresh and the sibling
  Leto/Eunomia `Quantity<T>::in_unit` / `FloatElement` mismatch; hosted
  checkout-graph CI is the accepted source/build gate.
  Merged through PR #94 after the final documentation-head rerun.

## HEPH-BACKEND-PARITY-STENCIL-1 [minor] — done

- Owner: Codex; scope: shared 2D Laplacian contract in `hephaestus-core`,
  native CUDA and ROCm stencil kernels, Metal delegation, value-semantic
  backend contracts, existing CUDA/ROCm/Metal CI lanes, and synchronized
  backend documentation. Prepared reductions, sparse batches, and remaining
  decomposition or linalg differences are non-goals for this increment.
- Acceptance: WGPU, CUDA, ROCm, and Metal expose the same
  `Laplacian2DParams`/`Laplacian2DKernel` contract; native CUDA and HIP
  execute the device-resident 2D Laplacian with Dirichlet, Neumann, and
  periodic boundaries plus both polarity conventions; input/output storage
  and grid validation are typed and value-tested; Metal delegates through
  its native WGPU-Metal device; and each backend's CI lane runs the focused
  contracts. Hosted feature lanes passed at head `0718d6a`: CUDA run
  `30170135462` / job `89709752625`, ROCm run `30170135447` / job
  `89709752628`, and Metal run `30170135476` / job `89709752732`. Hardware
  lanes were skipped because hosted GPU labels were unavailable; required-
  device enforcement remains enabled for self-hosted GPU runners. Local
  Windows package compilation remains blocked by the sibling Leto/Eunomia
  `Quantity<T>::in_unit` / `FloatElement` mismatch. Merged through PR #93.
- Claimed files: `crates/hephaestus-core/src/domain/stencil.rs`, the four
  backend stencil modules and exports/tests, `Cargo.toml` dependency entries,
  `README.md`, `CHANGELOG.md`, `docs/adr/0014-backend-stencil-parity.md`,
  `checklist.md`, and this item. Last update: 2026-07-25.

## HEPH-BACKEND-PARITY-VOLUME-1 [minor] — done

- Owner: Codex; scope: shared `FieldGeometry`/ray-packing contract in
  `hephaestus-core`, native CUDA and ROCm volume ray-integral kernels, Metal
  delegation, value-semantic contracts, CUDA/ROCm/Metal CI, and synchronized
  backend documentation. Prepared reductions, the Laplacian stencil, and
  other backend-only families are non-goals for this increment.
- Acceptance: WGPU, CUDA, ROCm, and Metal expose the same
  `FieldGeometry`/`RAY_STRIDE`/`ray_line_integrals(_into)` contract; CUDA and
  HIP execute the midpoint trilinear ray integral on device-resident field and
  ray buffers; misses, empty output, invalid step, length mismatch, field-size
  mismatch, and exact-f32 count limits are value-tested; Metal delegates the
  same contract through its native WGPU-Metal device; and each backend's CI
  lane builds and runs the focused contract with required-device enforcement
  where hardware is available.
- Claimed files: `crates/hephaestus-core/src/domain/volume.rs`, the four
  backend volume modules and exports/tests, CUDA/ROCm/Metal workflow files,
  `README.md`, `CHANGELOG.md`, `docs/adr/0013-backend-volume-parity.md`,
  `checklist.md`, and this item. Last update: 2026-07-25. Hosted feature
  lanes passed at head `b7c81fb`: CUDA run `30167708589` / job
  `89703450676`, ROCm run `30167708590` / job `89703450758`, and Metal run
  `30167708592` / job `89703450752`. Hardware lanes were skipped because the
  hosted runner labels were unavailable; required-device enforcement remains
  enabled for self-hosted GPU runners. Local Windows package compilation
  remains blocked by the sibling Leto/Eunomia `Quantity<T>::in_unit` /
  `FloatElement` mismatch.

## HEPH-ROCM-PARITY-CHOLESKY-1 [minor] — done

- Owner: Codex; scope: ROCm decomposition feature seam, device-resident
  Cholesky factorization and blocked entry point, value-semantic contracts,
  decomposition feature CI, and synchronized backend documentation. LU, QR,
  eigen, SVD, and other decomposition families are non-goals for this
  increment.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuCholesky`, `cholesky_decompose`, and `cholesky_decompose_blocked` surface;
  factorization executes through real HIP kernels with typed failure reporting;
  empty, non-square, non-finite, non-positive-definite, dense, and strided
  contracts are value-tested; and hosted ROCm feature CI runs the decomposition
  feature through build, warning-denied Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/Cargo.toml`,
  `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/application/mod.rs`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `.github/workflows/rocm.yml`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm run `30127038908` passed the base and
  `rocm,decomposition` checks, warning-denied Clippy, Nextest (38/38),
  doctest, and rustdoc at head `cb657e3`; the required-device lane was skipped
  by the pull-request event. Merged through PR #80 at `4f52769`. Local package
  compilation remains blocked by the sibling Leto checkout's
  `Quantity<T>::in_unit` / `FloatElement` mismatch.

## HEPH-ROCM-PARITY-DECOMPOSITION-2 [minor] — done

- Owner: Codex; scope: ROCm LU and QR decomposition surfaces, real HIP
  factorization kernels, device-resident factors, solve/determinant/inverse or
  least-squares contracts where the common backends expose them, value-semantic
  tests, decomposition feature CI, and synchronized backend documentation.
  SVD, eigen, Schur, Hessenberg, bidiagonal, UDU, Bunch–Kaufman, complete-pivot
  LU, and column-pivoted QR remain non-goals for this increment.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuLuDecomposition` and `GpuQrDecomposition` APIs; factorization runs on HIP
  for dense and strided entry points; invalid, empty, singular, and
  rank-deficient contracts return typed results; LU/QR value contracts compare
  device results with the independent `leto-ops` reference; and hosted ROCm CI
  passes build, warning-denied Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/Cargo.toml`,
  `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/application/mod.rs`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `.github/workflows/rocm.yml`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm run `30129776247` passed the base and
  `rocm,decomposition` checks, warning-denied Clippy, Nextest, doctest, and
  rustdoc at head `47e13ee`; the required-device lane was skipped by the
  pull-request event. Merged through PR #81 at `82d3bc8`.

## HEPH-ROCM-PARITY-PIVOTED-3 [minor] — done

- Owner: Codex; scope: ROCm complete-pivot LU and column-pivoted QR surfaces,
  real HIP pivot/factorization kernels, device-resident factors and
  permutations, solve/inverse or least-squares contracts, value-semantic
  tests, decomposition feature CI, and synchronized backend documentation.
  SVD, eigen, Schur, Hessenberg, bidiagonal, UDU, and Bunch–Kaufman remain
  non-goals for this increment.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuFullPivLuDecomposition`/`full_piv_lu` and
  `GpuColPivQrDecomposition`/`col_piv_qr` APIs; factorization executes with
  HIP kernels for the supported dense and strided contracts; row/column
  permutations, rank, solve, inverse, and least-squares values match the
  independent `leto-ops` reference; invalid, empty, singular, and
  rank-deficient cases return typed results; and hosted ROCm CI passes build,
  warning-denied Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/application/pipeline.rs`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `.github/workflows/rocm.yml`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm run `30131762591` passed the base and
  `rocm,decomposition` checks, warning-denied Clippy, Nextest, doctest, and
  rustdoc at head `d030520`; the required-device lane was skipped by the
  pull-request event. Merged through PR #82 at `6613690`.

## HEPH-ROCM-PARITY-SPECTRAL-4 [minor] — done

- Owner: Codex; scope: ROCm bidiagonalization and SVD surfaces, device-resident
  U/B/V and U/V/singular-value buffers, thin/rank-revealing/singular-values
  contracts, value-semantic reconstruction and rank tests, decomposition
  feature CI, and synchronized backend documentation. Eigen, Schur,
  Hessenberg, UDU, Bunch–Kaufman, and other decomposition families remain
  non-goals for this increment.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuBidiagonalDecomposition`/`bidiagonalize` and
  `GpuSvdDecomposition`/`svd_decompose`/`svd_rank_revealing`/`singular_values`
  APIs; tall-or-square and empty/invalid contracts are typed; ROCm uploads the
  shared `leto-ops` provider results into typed device buffers without a
  backend-selection fallback; and hosted ROCm CI passes build, warning-denied
  Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `README.md`, `CHANGELOG.md`, `docs/adr/0012-rocm-backend.md`, `checklist.md`,
  and this item. Last update: 2026-07-24. Hosted ROCm run `30132885402`
  passed base and `rocm,decomposition` builds, warning-denied Clippy, Nextest,
  doctest, and rustdoc at head `6864aa7`; the required-device lane was skipped
  by the pull-request event. Merged through PR #83 at `56e83e4`.

## HEPH-ROCM-PARITY-SYMMETRIC-5 [minor] — done

- Owner: Codex; scope: ROCm UDU and Bunch–Kaufman decomposition surfaces,
  device-resident U/D and L/D/permutation results, provider-backed solve,
  determinant, inverse, reconstruction, empty, invalid, and permutation
  contracts, decomposition feature CI, and synchronized documentation. Eigen,
  Schur, Hessenberg, and other decomposition families remain non-goals for
  this increment.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuUduDecomposition`/`udu_decompose` and
  `GpuBunchKaufmanDecomposition`/`bunch_kaufman` APIs; ROCm uploads the shared
  Leto factors into typed device buffers without a backend-selection fallback;
  UDU solve/determinant/inverse and Bunch–Kaufman permutation/reconstruction
  behavior are value-tested; and hosted ROCm CI passes build, warning-denied
  Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `README.md`, `CHANGELOG.md`, `docs/adr/0012-rocm-backend.md`, `checklist.md`,
  and this item. Last update: 2026-07-24. Hosted ROCm run `30134286942`
  passed base and `rocm,decomposition` builds, warning-denied Clippy, Nextest,
  doctest, and rustdoc at head `8304d07`; the required-device lane was skipped
  by the pull-request event. Merged through PR #84 at `fce147d`.

## HEPH-ROCM-PARITY-EIGEN-6 [minor] — done

- Owner: Codex; scope: ROCm symmetric Jacobi eigenpairs/eigenvalues and general
  complex eigenvalues, device-resident typed result buffers, provider-backed
  reconstruction and spectrum contracts, empty/invalid/non-finite validation,
  decomposition feature CI, and synchronized documentation.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuSymmetricEigenDecomposition`/`symmetric_eigen_jacobi`,
  `symmetric_eigenvalues_jacobi`, and `eigenvalues` APIs; ROCm uploads shared
  Leto results into typed f32/complex buffers without a backend-selection
  fallback; symmetric reconstruction and general complex spectra are
  value-tested; and hosted ROCm CI passes build, warning-denied Clippy,
  Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/Cargo.toml`,
  `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `README.md`, `CHANGELOG.md`, `docs/adr/0012-rocm-backend.md`, `checklist.md`,
  and this item. Hosted ROCm run `30134785595` passed the base and
  `rocm,decomposition` checks, warning-denied Clippy, Nextest, doctest, and
  rustdoc at head `8e4eaf8`; the required-device lane was skipped by the
  pull-request event. Merged through PR #85 at `b122474`.

## HEPH-ROCM-PARITY-SCHUR-7 [minor] — done

- Owner: Codex; scope: ROCm Hessenberg reduction and real Schur decomposition,
  device-resident typed Q/H and Q/T buffers, provider-backed similarity and
  structure contracts, empty/invalid/non-finite validation, decomposition
  feature CI, and synchronized documentation. No further common CUDA/WGPU
  decomposition families are known after this increment.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `GpuHessenbergDecomposition`/`hessenberg` and `GpuRealSchur`/`schur` APIs;
  ROCm uploads shared Leto factors into typed device buffers without a
  backend-selection fallback; orthogonality, reconstruction, Hessenberg and
  real-Schur structure, spectra, empty, rectangular, and non-finite cases are
  value-tested; and hosted ROCm CI passes build, warning-denied Clippy,
  Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/src/application/decomposition/**`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `README.md`, `CHANGELOG.md`, `docs/adr/0012-rocm-backend.md`, `checklist.md`,
  and this item. Hosted ROCm run `30135505585` passed the base and
  `rocm,decomposition` builds, warning-denied Clippy, Nextest (62/62),
  doctest, and rustdoc at implementation head `6ea7a0c`; the required-device
  lane was skipped by the pull-request event. PR #86 is the merge vehicle.

## HEPH-ROCM-PARITY-MATRIX-8 [minor] — done

- Owner: Codex; scope: ROCm pseudoinverse and matrix-exponential surfaces,
  provider-backed typed result buffers, Moore–Penrose/closed-form contracts,
  empty/invalid/non-finite validation, decomposition feature CI, and
  synchronized documentation.
- Acceptance: enabling `rocm,decomposition` exposes the common CUDA/WGPU
  `pinv` and `matexp` APIs; ROCm uploads shared Leto results into typed device
  buffers without a backend-selection fallback; diagonal, rectangular,
  rank-deficient, Moore–Penrose, closed-form, general, empty, square, and
  non-finite cases are value-tested; and hosted ROCm CI passes build,
  warning-denied Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/src/application/linalg/**`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `README.md`, `CHANGELOG.md`, `docs/adr/0012-rocm-backend.md`, `checklist.md`,
  and this item. Hosted ROCm run `30136255296` passed the base and
  `rocm,decomposition` builds, warning-denied Clippy, Nextest (65/65),
  doctest, and rustdoc at implementation head `f67da57`; the required-device
  lane was skipped by the pull-request event. PR #87 is the merge vehicle.

## HEPH-ROCM-PARITY-FLUENT-9 [minor] — done

- Owner: Codex; scope: ROCm fluent matrix traits for operand conversion,
  products, norms, decomposition, solves, matrix properties, and matrix
  functions, plus value-semantic contract coverage and synchronized backend
  documentation. Backend-specific prepared kernels and dynamic-rank helpers
  remain non-goals unless the common CUDA/WGPU public surface requires them.
- Acceptance: enabling `rocm` and `rocm,decomposition` exports the same fluent
  `AsGpuMatrixOperand`, `MatrixProduct`, `MatrixNorm`, `MatrixDecompose`,
  `MatrixSolve`, `MatrixProperties`, and `MatrixFunction` contracts as CUDA
  and WGPU; each method delegates to the already-parity-tested ROCm operation
  family; trait calls preserve typed validation and device-resident outputs;
  value-semantic tests exercise representative product, norm, decomposition,
  solve, property, and function calls; and hosted ROCm CI passes build,
  warning-denied Clippy, Nextest, doctest, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/src/application/linalg/**`,
  `crates/hephaestus-rocm/src/lib.rs`, `crates/hephaestus-rocm/tests/contract.rs`,
  `README.md`, `CHANGELOG.md`, `docs/adr/0012-rocm-backend.md`, `checklist.md`,
  and this item. Hosted ROCm run `30137841431` passed the base and
  `rocm,decomposition` builds, warning-denied Clippy, Nextest (67/67),
  doctest, and rustdoc at implementation head `356c610`; the required-device
  lane was skipped by the pull-request event. PR #88 is the merge vehicle.

## HEPH-ROCM-PARITY-EXPORTS-10 [minor] — done

- Owner: Codex; scope: ROCm crate-root exports for the shared elementwise
  operation marker types and their compile-time public-surface contract.
- Acceptance: ROCm root exports the same shared unary and binary operation
  marker types as CUDA and wgpu; the contract compiles through the public
  root path; hosted ROCm CI passes the affected feature matrix and tests.
- Claimed files: `crates/hephaestus-rocm/src/lib.rs`,
  `crates/hephaestus-rocm/tests/contract.rs`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Hosted ROCm
  run `30138469322` passed the base and `rocm,decomposition` builds,
  warning-denied Clippy, Nextest (68/68), doctest, and rustdoc at
  implementation head `5031196`; the required-device lane was skipped by the
  pull-request event. PR #89 merged as `496ff8c`. The external
  `recurseml/analysis` check reported an analyzer-side error and was not a
  required check.

## HEPH-ROCM-PARITY-STREAM-1 [minor] — done

- Owner: Codex; scope: ROCm implementations of `KernelDevice`,
  `CommandStream`, `GroupedKernelDevice`, and grouped sequencing, including
  real HIP module launches, device copies, prefix copies, byte fills, and
  value-semantic contracts. Operator-family parity beyond authored-kernel
  dispatch is a non-goal for this increment.
- Acceptance: ROCm exposes prepared and grouped prepared kernel contracts
  equivalent to CUDA/WGPU, validates bindings through `hephaestus-core`, keeps
  ordered dispatch/copy/fill operations on HIP's default stream, and hosted
  ROCm feature CI passes with warning-denied Clippy, Nextest, doctests, and
  rustdoc.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Merged through PR #79 at `cd5a699`; valid hosted ROCm container
  run `30125278305` passed the real feature build, warning-denied Clippy,
  Nextest (34/34), doctest, and rustdoc at head `3ac1a22`; the required-device
  lane was skipped by `run_hardware=false`.

## HEPH-ROCM-PARITY-STORAGE-1 [minor] — done

- Owner: Codex; scope: ROCm implementations of the existing
  `MultiStorageKernel`/`MultiStorageDevice` seams, real HIP module launches,
  pre-launch layout and length validation, value-semantic contracts, and the
  existing ROCm CI lanes. Authored-kernel command streams are a non-goal for
  this increment.
- Acceptance: ROCm exposes typed `RocmStorageBinding` and
  `RocmMultiStorageKernel` contracts equivalent to the CUDA/WGPU multi-storage
  surface; HIP launches real unary/binary storage kernels; invalid bindings,
  dimensions, and lengths fail before launch; and hosted ROCm feature CI passes
  with warning-denied Clippy, Nextest, doctests, and rustdoc.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Merged through PR #79 at `cd5a699`; valid hosted ROCm container
  run `30125278305` passed the real feature build, warning-denied Clippy,
  Nextest (34/34), doctest, and rustdoc at head `3ac1a22`; the required-device
  lane was skipped by `run_hardware=false`.

## HEPH-ROCM-PARITY-SPARSE-1 [minor] — done

- Owner: Codex; scope: ROCm device-resident CSR storage, HIP SpMV/SpMM and
  multi-RHS SpMV reuse, value-semantic CPU contracts, and the existing ROCm CI
  lanes. Streams and storage families are non-goals for this increment.
- Acceptance: ROCm exposes the CUDA/WGPU CSR `GpuCsrMatrix`, `spmv`, `spmv_into`,
  `spmm`, `spmm_into`, `spmv_many`, and `spmv_many_into` contracts; CSR metadata
  remains typed device storage; HIP kernels compute real sparse products; and
  round-trip, value, reuse, and shape contracts pass in the container and
  required-device lanes.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Merged through PR #79 at `cd5a699`; valid hosted ROCm container
  run `30125278305` passed the real feature build, warning-denied Clippy,
  Nextest (34/34), doctest, and rustdoc at head `3ac1a22`; the required-device
  lane was skipped by `run_hardware=false`.

## HEPH-ROCM-PARITY-RANDOM-1 [minor] — done

- Owner: Codex; scope: ROCm seeded uniform and normal initializers using the
  shared deterministic `leto-ops` contract, typed device uploads, CPU-value
  contracts, and the existing ROCm CI lanes. Sparse, streams, and storage
  families are non-goals for this increment.
- Acceptance: ROCm exposes the same `uniform_with_seed` and
  `normal_with_seed` contracts as CUDA and WGPU for supported real scalars and
  ranks, preserves deterministic repeated seeds and uniform bounds, maps
  producer errors to typed backend errors, and uploads the resulting values to
  typed ROCm buffers. The container lane compiles and tests the feature path,
  while the required-device lane executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Hosted ROCm
  container run `30119890105` passed the real feature build, warning-denied
  Clippy, Nextest (28/28), doctest, and rustdoc at corrected PR head `4005991`;
  PR #78 merged as `81bed23`. The required-device lane remained skipped for the
  pull-request event.

## HEPH-ROCM-PARITY-MATRIX-PROPERTIES-1 [minor] — done

- Owner: Codex; scope: ROCm finite matrix rank estimation and determinant over
  strided rank-2 inputs using one HIP Gaussian-elimination kernel, shared
  layout validation, CPU differential contracts, and the existing ROCm CI
  lanes. Sparse, streams, storage, and random families are non-goals for this
  increment.
- Acceptance: ROCm exposes the same `matrix_rank`,
  `matrix_rank_with_tolerance`, and `det` contracts as CUDA and WGPU for
  rank-2 views, validates finite non-negative tolerance, shape, storage, and
  offsets, computes real device-side partial-pivot elimination, returns zero
  for singular determinants, and matches CPU values. The container lane
  compiles and tests the real feature path, while the required-device lane
  executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm container run `30118716147` passed the real feature
  build, warning-denied Clippy, Nextest (27/27), doctest, and rustdoc at PR
  head `ca602c6`; PR #77 merged as `148436f`. The required-device lane
  remained skipped for the PR event.

## HEPH-ROCM-PARITY-MATPOW-1 [minor] — done

- Owner: Codex; scope: ROCm matrix powers over strided rank-2 inputs using
  native identity-copy, tiled matrix multiplication, exponentiation by
  squaring, CPU differential contracts, and the existing ROCm CI lanes.
  Matrix properties, sparse, streams, storage, and random families are
  non-goals for this increment.
- Acceptance: ROCm exposes the same `matpow` contract as CUDA and WGPU for
  square rank-2 views, validates square shape and storage, returns identity
  for exponent zero, computes real device-resident products without CPU or
  WGPU fallback, and returns CPU-reference values for contiguous and strided
  inputs. The container lane compiles and tests the real feature path, while
  the required-device lane executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm container run `30117981372` passed the real feature
  build, warning-denied Clippy, Nextest (25/25), doctest, and rustdoc at PR
  head `cd31db6`; PR #76 merged as `809c79f`. The required-device lane
  remained skipped for the PR event.

## HEPH-ROCM-PARITY-STRIDED-1 [minor] — done

- Owner: Codex; scope: ROCm rank-≤4 strided binary, unary, and scalar
  elementwise dispatch using one packed HIP metadata/decode core, broadcast
  layouts, caller-owned and allocating APIs, CPU differential contracts, and
  the existing ROCm CI lanes. Matrix power, matrix properties, sparse,
  streams, storage, and random families are non-goals for this increment.
- Acceptance: ROCm exposes the same strided elementwise operation contracts as
  CUDA and WGPU for supported rank-≤4 Leto layouts, validates broadcast shape,
  storage, offsets, signed strides, output zero-stride aliasing, and buffer
  aliasing before launch, computes real HIP kernels for binary, unary, and
  scalar operations, and returns CPU-reference values. The container lane
  compiles and tests the real feature path, while the required-device lane
  executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm container run `30117107666` passed the real feature
  build, warning-denied Clippy, Nextest (23/23), doctest, and rustdoc at PR
  head `8699345`; PR #75 merged as `8060f33`. The required-device lane remained
  skipped for the PR event.

## HEPH-ROCM-PARITY-KRON-1 [minor] — done

- Owner: Codex; scope: ROCm strided rank-2 Kronecker product and
  caller-owned output using one HIP coordinate-mapping kernel, shared matrix
  layout metadata, typed outputs, CPU differential contracts, and the existing
  ROCm CI lanes. Matrix properties, sparse, strided elementwise, streams,
  storage, and random families are non-goals for this increment.
- Acceptance: ROCm exposes the same `kron`/`kron_into` contract as CUDA and
  WGPU for rank-2 strided views, validates output shape, storage, offset,
  stride, zero-stride aliasing, and buffer aliasing before launch, computes the
  real HIP product for non-multiple output extents, and returns CPU-reference
  values. The container lane compiles and tests the real feature path, while
  the required-device lane executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm container run `30115666613` passed the real feature
  build, warning-denied Clippy, Nextest (21/21), doctest, and rustdoc at PR
  head `16cead2`; PR #74 merged as `6ee586f`. The required-device lane
  remained skipped for the PR event.

## HEPH-ROCM-PARITY-NORMS-1 [minor] — done

- Owner: Codex; scope: ROCm strided dot product, trace, L1, L2/Frobenius,
  and max-magnitude norms using one HIP map-reduction kernel, shared rank-four
  layout metadata, typed outputs, CPU differential contracts, and the existing
  ROCm CI lanes. Kronecker, matrix properties, sparse, strided elementwise,
  streams, storage, and random families are non-goals for this increment.
- Acceptance: ROCm exposes the same dot/trace/norm contracts as CUDA and WGPU
  for rank-1/rank-2/rank-N strided views, validates shape, storage, offset,
  stride, and empty-input boundaries before launch, computes real HIP
  map-reductions plus square-root completion for L2, and returns CPU-reference
  values. The container lane compiles and tests the real feature path, while
  the required-device lane executes the same contracts on AMD hardware.
- Claimed files: crates/hephaestus-rocm/**, README.md, CHANGELOG.md,
  docs/adr/0012-rocm-backend.md, checklist.md, and this item. Last update:
  2026-07-24. Hosted ROCm container run `30114338471` passed the real feature
  build, warning-denied Clippy, Nextest (19/19), doctest, and rustdoc at PR
  head `32dc87c`; PR #73 merged as `fb702e9`. The required-device lane
  remained skipped for the PR event.

## HEPH-ROCM-PARITY-BATCHED-MATMUL-1 [minor] — done

- Owner: Codex; scope: ROCm rank-3 batched matrix multiplication using the
  shared tiled matmul kernel family, singleton-batch broadcasting, checked
  batch strides, typed allocating and caller-owned output APIs, CPU
  differential contracts, and the existing ROCm CI lanes. Norms, Kronecker,
  matrix properties, sparse, strided elementwise, streams, storage, and random
  families are non-goals for this increment.
- Acceptance: ROCm exposes the same rank-3 `batched_matmul`/`batched_matmul_into`
  contract as CUDA and WGPU, validates batch shape, storage, layout, output
  aliasing, and zero-stride output races before launch, computes partial tiles,
  broadcasts singleton inputs, and returns CPU-reference values. The container
  lane compiles and tests the real feature path, while the required-device lane
  executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Last update:
  2026-07-24. Hosted ROCm container run `30112489093` passed the real feature
  build, warning-denied Clippy, Nextest (17/17), doctest, and rustdoc at PR
  head `5377733`; PR #72 merged as `2634776`. The required-device lane
  remained skipped for the PR event.

## HEPH-ROCM-PARITY-MATMUL-1 [minor] — done

- Owner: Codex; scope: ROCm rank-2 tiled matrix multiplication using shared
  strided layouts, HIP module launches, typed allocating and caller-owned
  output APIs, CPU differential contracts, and the existing ROCm CI lanes.
  Batched linalg, sparse, strided elementwise, streams, storage, and random
  families are non-goals for this increment.
- Acceptance: ROCm exposes the same rank-2 `matmul`/`matmul_into` contract as
  CUDA and WGPU, validates shape, storage, layout, and output aliasing before
  launch, computes partial 16×16 edge tiles correctly, and returns values
  matching a CPU reference. The container lane compiles and tests the real
  feature path, while the required-device lane executes the same contracts on
  AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Hosted ROCm
  container run `30111559905` passed the real feature build, warning-denied
  Clippy, Nextest (15/15), doctest, and rustdoc at PR head `74c9948`; PR #71
  merged as `29e8e5c`. The required-device lane remained skipped for the PR
  event. Last update: 2026-07-24.

## HEPH-ROCM-PARITY-SCAN-1 [minor] — done

- Owner: Codex; scope: ROCm rank-2 forward/reverse scans using the shared
  `AxisScanMeta` and `plan_axis_scan` contracts, HIP module launches over typed
  strided operands, cumulative sum/product APIs, value-semantic CPU
  differential tests, and the existing ROCm CI lanes. Linalg, sparse, strided
  elementwise, streams, storage, and random families are non-goals for this
  increment.
- Acceptance: ROCm exposes the same rank-2 scan contract as CUDA and WGPU,
  preserving shape and direction semantics for cumulative sum/product,
  validating shape/stride/storage/alias boundaries through the shared core
  planner, and returning CPU-reference values across axes, directions, and
  tiled long lines. The container lane compiles and tests the real feature
  path, while the required-device lane executes the same contracts on AMD
  hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Hosted ROCm
  container run `30109934133` passed the corrected scan feature build,
  warning-denied Clippy, Nextest (13/13), doctest, and rustdoc at PR head
  `3d70841`; PR #70 merged as `06dd503`. The required-device lane remained
  skipped for the PR event. Last update: 2026-07-24.

## HEPH-ROCM-PARITY-AXIS-REDUCTION-1 [minor] — done

- Owner: Codex; scope: ROCm rank-2 axis sum/min/max/mean reductions using the
  shared `AxisReductionMeta` and `plan_axis_reduction` contracts, HIP module
  launches over typed strided operands, value-semantic CPU differential tests,
  and the existing ROCm CI lanes. Scans, sparse, linalg, streams, storage,
  and random families are non-goals for this increment.
- Acceptance: ROCm exposes the same rank-2 axis-reduction contract as CUDA and
  WGPU, preserving the reduced axis as length one, validating shape/stride/
  alias/empty-axis boundaries through the shared core planner, and returning
  values matching CPU references for sum, min, max, and mean. The container
  lane compiles and tests the real feature path, while the required-device lane
  executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Hosted ROCm
  container run `30108405040` passed the real feature build, warning-denied
  Clippy, Nextest (11/11), doctest, and rustdoc at PR head `04dcce4`; PR #69
  merged as `ab4b407`. The required-device lane remained skipped for the PR
  event. Last update: 2026-07-24.

## HEPH-ROCM-PARITY-REDUCTION-1 [minor] — done

- Owner: Codex; scope: ROCm 1D sum/min/max reduction kernels using the shared
  `HipC` operation and identity vocabulary, multi-pass typed-buffer ownership,
  value-semantic CPU differential contracts, and the existing ROCm CI lanes.
  Axis reductions, scans, sparse, linalg, streams, storage, and random
  families are non-goals for this increment.
- Acceptance: ROCm exposes the same contiguous 1D reduction contract as CUDA
  and WGPU through the shared operation markers; HIP kernels compile through
  hipRTC, load through the HIP module API, reduce non-empty and empty inputs
  over multiple passes, reject invalid widths and representable-length
  violations, and return values matching the CPU oracle. The container lane
  compiles and tests the real feature path, while the required-device lane
  executes the same contracts on AMD hardware.
- Claimed files: `crates/hephaestus-rocm/**`, `README.md`, `CHANGELOG.md`,
  `docs/adr/0012-rocm-backend.md`, `checklist.md`, and this item. Hosted ROCm
  container run `30106758162` passed the real feature build, warning-denied
  Clippy, Nextest (10/10), doctest, and rustdoc at PR head `621bbf2`; PR #68
  merged as `1146ee4`. The required-device lane remained skipped for the PR
  event. Last update: 2026-07-24.

## HEPH-ROCM-PARITY-ELEMENTWISE-1 [minor] — done

- Owner: Codex; scope: shared HIP-C dialect vocabulary, ROCm runtime-compiled
  elementwise binary/unary/scalar kernels, value-semantic ROCm contracts, and
  the existing ROCm CI build and hardware lanes. Reductions, scans, sparse,
  linalg, streams, storage, and random families are non-goals for this
  increment.
- Acceptance: ROCm exposes the same elementwise operation contract as CUDA
  through the shared operation markers; HIP kernels compile through hipRTC,
  load through the HIP module API, launch over typed device buffers, reject
  invalid lengths and output aliasing, and return values matching the CPU
  oracle. The container lane compiles and tests the real feature path, while
  the required-device lane executes the same tests on AMD hardware.
- Claimed files: `crates/hephaestus-core/src/domain/{dialect,ops}.rs`,
  `crates/hephaestus-rocm/**`, `.github/workflows/rocm.yml`,
  `docs/adr/0012-rocm-backend.md`, `README.md`, `CHANGELOG.md`, `checklist.md`,
  and this item. Hosted ROCm container run `30105156934` passed the real
  feature checks, warning-denied Clippy, Nextest (9/9), doctest, and rustdoc at
  PR head `563783f`; the required-device lane remained skipped for the PR
  event. PR #67 merged as `7e1fbb9`. Last update: 2026-07-24.

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
  parity done; blocked-decomposition host loops and wrappers remain. Claimed
  2026-08-02 by user session: blocked LU host-loop slice first, per ADR-0003
  sequencing (LU → QR → Cholesky). **Blocked-LU host loop hoisted**
  (2026-08-02, PR #189): `blocked_lu` over the new `BlockedDecompositionBackend`
  trait in `hephaestus-core`; wgpu + cuda entry points reduced to thin
  delegators; cuda `download_region` fills a caller `&mut Vec` per ADR-0003.
  Gates green (fmt, clippy -D warnings core/wgpu/cuda all-features, core 84/84,
  wgpu + cuda blocked_lu 5/5 each). Remaining per ADR-0003: QR and Cholesky
  loop-structure trait hoists. The O(L²)
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
  Status: done (verified 2026-08-02, owner user session). The 12-leaf-module
  split (`2deb976`), `split_packed_lu` eviction (`6a99625`), and the residual
  audit/eviction (`4c6f89e`) are on origin/master. Eviction delivered in
  `4c6f89e`: `ColPivQrHandle::shape()` seam accessor (core + wgpu/cuda/rocm/
  metal + conformance shape clause) replaces the python host-side
  `sqrt`/`checked_div` reconstruction of `col_piv_qr` shapes; python crate is
  now ~98% thin binding surface. Kept by design (documented in `4c6f89e`):
  `mean` reciprocal and `__rpow__` `ln` host scalars (irreducible without new
  backend kernels; no full-array mean op exists) and the PyO3-boundary shape
  validation (different error-domain binding helpers, not duplicated math).
  Incidental findings: themis git-package resolution defect (workspace
  declared git package `themis`, which upstream renamed to `themis-topology`
  while keeping lib name `themis`; the committed no-source Cargo.lock entry
  masked it under `--locked` — any re-resolution failed, including CI's
  `cargo update` step). Fixed in `473dcf7`: `package = "themis-topology"` on
  the workspace dependency; extern crate name unchanged. Pre-existing
  `clippy::collapsible_if` debt in
  `hephaestus-conformance/src/transfer.rs:119` remains (CI never clippies
  conformance).
- [KS-7] [minor] Perf batch from the audit: CUDA streams + pinned staging
  (CU-P1/P6/M3), batched-matmul `blockIdx.z` (CU-P5), typed CUDA cache keys
  (CU-P9/P10), wgpu encoder-borrowing batching (WG-P4), fused dot/norms
  (WG-P3), rank/det serial-kernel fix (WG-P1), axis-1 grid-stride reduction
  (WG-P5). Status: in-progress (owner codex; scope `hephaestus-wgpu/src/
  application/{bindings,stream,strided,storage_kernel}.rs`, CUDA/ROCm
  `application/storage_kernel.rs`, and manifests); criterion baselines
  before/after each.
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
  - **CU-P11 closed locally**: direct CUDA `dot`, `trace`, `norm_l1`,
    `norm_l2`, and `norm_max` now use a fused strided map-reduction kernel,
    retaining only `ceil(logical_len / BlockWidth::DEFAULT)` workgroup partials
    instead of a full-length elementwise scratch result. The direct path has
    contiguous and reversed-view value tests plus a comparative benchmark
    instrument. Local evidence: no-feature Nextest 78/78, CUDA feature check,
    CUDA/decomposition all-target check, feature-gated clippy, formatting, and
    doctests. Physical CUDA execution remains CI/self-hosted evidence.
  - **CU-P12 closed locally**: prepared CUDA and ROCm dot/L2-norm plans now
    reuse the fused first-pass map-reduction object across dispatches. Each
    plan retains `ceil(logical_len / BlockWidth::DEFAULT)` workgroup partials,
    an optional prepared reduction tree, and the stable scalar output instead
    of allocating a full logical-length product/square buffer. Existing
    repeated-dispatch, value, layout, and allocation contracts remain in
    place. Local evidence: CUDA feature/decomposition all-target check,
    CUDA feature Clippy, CUDA feature doctests, CUDA no-feature Nextest 78/78,
    ROCm no-feature Clippy, ROCm no-feature doctests, and ROCm no-feature
    Nextest 49/49. The ROCm feature build remains a Linux-only CI gate because
    this Windows checkout intentionally rejects it without a ROCm installation;
    physical CUDA/ROCm execution remains CI/self-hosted evidence.
  - **CU-P13 closed locally**: CUDA authored command streams now retain one
    bounded launch-scratch pair per command stream and reuse it across direct
    encodes, grouped encodes, and grouped sequences. The backend-neutral stream
    contract and legacy-null-stream ordering are unchanged. The new unit
    contract verifies capacity reuse after growth; adapterless Nextest is
    79/79, feature/all-target check and feature Clippy are warning-clean,
    no-feature doctests and formatting pass. Feature Nextest cannot link in
    this Windows GNU checkout because `-lcuda` is unavailable; CI remains the
    feature-linked and physical-device gate. No runtime speedup claim is made.
  - **RC-P1 closed locally**: ROCm authored command streams now retain and
    reuse one bounded launch-scratch pair for direct, grouped, and
    grouped-sequence encodes, matching CU-P13's CUDA path. The new capacity
    contract passes; local no-feature Clippy, all-target check, Nextest 50/50,
    doctests, formatting, and diff checks pass. The ROCm feature-linked lane
    remains a Linux ROCm CI gate from this Windows checkout.
  - **WG-P7 closed locally**: WGPU command streams retain uniform-buffer
    lifetime guards in inline-capacity storage for the common one-to-four
    dispatch case, spilling for longer streams without changing submission
    order or buffer recycling. Acceptance: common streams remain heap-free at
    the container layer; grouped and direct stream tests plus WGPU CI remain
    green. No runtime speedup claim is made without a device benchmark.
  - **WG-P6/CU-P14/RC-P2 closed locally**: WGPU strided, authored-stream, and
    multi-storage dispatches, plus direct CUDA and ROCm multi-storage
    dispatches, currently allocate host descriptor/argument vectors on every
    submission. Replace those vectors with inline-capacity storage that spills
    for larger kernels, preserving the existing binding and argument order.
    Acceptance: common one-to-four-resource WGPU descriptors and one-to-four
    pointer CUDA/ROCm launches remain heap-free at the container layer; larger
    binding counts retain the existing dynamic behavior; backend contract
    suites and adapterless checks remain green. Evidence: WGPU default-feature
    Nextest 160/160, CUDA/ROCm adapterless Nextest 129/129, WGPU/CUDA/ROCm
    clippy `-D warnings`, CUDA feature all-target check, doctests, formatting,
    and the inline-capacity regression all pass. No runtime speedup claim is
    made without a device benchmark.
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
- [KS-9] [minor] `hephaestus-metal` decision: retain the dedicated typed
  backend crate over wgpu-Metal. Status: **done** (2026-07-24). The crate owns
  `MetalDevice`/`MetalBuffer`, preserves the backend-neutral application
  boundary, and selects `wgpu::Backends::METAL` without exposing WGPU types to
  consumers. Collapsing it to `WgpuDevice::try_metal` would be a breaking
  public-surface change and would remove the Metal-specific required-device
  CI contract. The macOS workflow and README now carry the decision evidence.
  - **Revision 2026-08-14: superseded by Accepted ADR 0047**, which retires the
    crate as a WGPU adapter preference. Leaving this entry as `done` had the
    board asserting both positions at once. Two things were wrong here beyond
    the outcome. The crate does not "own" `MetalDevice`/`MetalBuffer` in any
    load-bearing sense - `MetalDevice` is a newtype around `WgpuDevice`
    acquired through `WgpuDevice::try_metal`, with zero native Metal API calls
    across 5 449 lines. And "would be a breaking public-surface change" is a
    prohibited tiebreaker: breaking-surface cost is what a `[major]` class
    exists to carry, not a reason to retain a forwarding layer.
    Retirement is tracked as ATLAS-ARCH-011 and is blocked on
    ATLAS-SUBSTRATE-002 (the `coeus-metal` consumer), not on anything in
    hephaestus - the removal was executed here, verified green, and reverted
    solely for that consumer.

- [KS-10] [major] `mnemosyne` co-evolution sweep: follow the upstream
  collision-free rename (`mnemosyne` package → `mnemosyne-memory`, Mnemosyne
  `79cd882`/`be4aa64`, 2026-08-02) in hephaestus's integration. Scope: update
  the CI checkout-local config template (`rocm-cargo-config.toml`) and the
  shared Atlas overlay to patch the renamed packages, align any workspace
  dependency declarations/features that name the pre-rename package, refresh
  the local `repos/mnemosyne` checkout (currently 5 behind, still exposing the
  stale `mnemosyne` package), and verify the graph resolves against the
  current remote without the no-source lockfile crutch. Acceptance: all four
  backend CI workflows reach their build/test steps (the configure step's
  `cargo update --workspace` currently fails with "does not contain packages
  matching `mnemosyne`"), and a fresh `cargo update --workspace` + full local
  gate passes. Blocker currently failing every hephaestus PR including the
  KS-6 lane's merged PR #180 (unblocked only because master is unprotected);
  the last green master CI was 2026-08-02T04:25Z, before the remote moved.
  Re-open trigger: none — this item is the cure.
  Status: done (verified 2026-08-02, user session). The CI configure
  template fix (`fix(ci)` 197b161/7f48f20/1a65172: patch `mnemosyne-memory` /
  `mnemosyne-memory-core` identities) plus the manifest/lockfile migration
  landed via merged PRs #179/#181/#182; `repos/mnemosyne` refreshed to
  `213fead`. Verification on the claimed lane (shared overlay + refreshed
  checkout): `cargo metadata --all-features` and `cargo update --workspace`
  (the CI configure step verbatim) exit 0; fmt/check (wgpu no-default-features
  lib + all-targets, cuda, metal)/clippy wgpu `-D warnings`/nextest core
  84/84/wgpu 210/210/doctest 1/1/doc all green. Local doc `--all-features`
  exits 101 only on `hephaestus-rocm`, the Linux-only `rocm` feature guard
  (CI-covered). Lockfile churn from the overlay refresh was intentionally
  reverted — the committed lock keeps git-source pins for non-overlay release
  builds.
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
