# Checklist — hephaestus

Sprint target: 0.18.0. Phase: Closure.

## HEPH-ROCM-SUBSTRATE-1 [arch]

- [x] Add the `hephaestus-rocm` workspace crate with a Linux-only optional
      `rocm` feature backed by the current HIP bindings.
- [x] Implement real HIP acquisition, typed allocation/transfer/sync,
      placement validation, capabilities, and Themis topology.
- [x] Add value-semantic contract tests for hardware and adapterless paths;
      require a device in the hardware CI lane.
- [x] Add ROCm container build/test CI plus manually enabled self-hosted AMD
      device CI, with path dependencies checked out at sibling locations.
- [x] Synchronize ADR, README, core contract docs, changelog, and evidence.
- [x] Pass formatting, locked check, warning-denied Clippy, configured
      Nextest, doctest, and rustdoc for the affected packages.

Local evidence (2026-07-24): `cargo fmt --all -- --check`, locked release
checks for the default and Linux `rocm` feature, warning-denied release
Clippy for both feature states, release Nextest 8/8 for the adapterless path,
doctest, rustdoc, metadata, and workflow YAML parsing pass. The local host has
no ROCm runtime or AMD device; these are not local hardware evidence.

Hosted evidence (2026-07-24): ROCm workflow run `30097596676` passes the
container build and verification lane in 5m34s at PR head `05300bc`; its
required-device AMD lane is correctly skipped for the pull-request event.
Hardware execution remains a separate workflow-dispatch acceptance gate on a
self-hosted runner labeled `rocm`.

Acceptance boundary: this increment owns the HIP device substrate only;
operator kernels are a follow-up item with an independent consumer contract.

## HEPH-PREPARED-MAP-REDUCTION-1 [minor]

- [x] Hoist reduction-tree encoding behind a caller-supplied encoder without
      duplicating pass planning or submission ownership.
- [x] Add prepared dot and L2-norm dispatch that reuse fixed-buffer resources
      and issue one command-buffer submission per operation.
- [x] Prove CPU-reference values, changed-input sensitivity, and stable output
      allocation identity on a real WGPU adapter.
- [x] Add and run the focused example plus a controlled prepared-versus-one-shot
      benchmark with unchanged inputs and Criterion measurement settings.
- [x] Pass focused format, warning-denied Clippy, Nextest, doctest, rustdoc,
      example, and benchmark gates; synchronize public documentation.

Current value evidence: `cargo nextest run -p hephaestus-wgpu -E
'test(prepared_dot_reuses_output_and_observes_input_updates) |
test(prepared_l2_norm_reuses_output_and_observes_input_updates)'
--no-fail-fast` passed 2/2 on a real adapter in 1.239 s. Both tests execute a
multi-pass reduction, rewrite the fixed input buffer, assert the changed exact
result, and assert the raw scalar output handle remains identical.

Isolated Criterion evidence on an Intel Core Ultra 9 285K / NVIDIA RTX 5080
(driver 610.47), 65,536 elements and 100 samples: dot one-shot
`[141.65, 144.79, 148.34] µs` versus prepared
`[105.19, 107.65, 110.40] µs` (25.7% lower point estimate); L2 one-shot
`[150.88, 158.89, 169.24] µs` versus prepared
`[119.79, 122.28, 124.79] µs` (23.0% lower point estimate). The first run was
rejected because it overlapped an Apollo benchmark and produced 14–18% severe
outliers; these figures are from the uncontended rerun with the instrument and
inputs unchanged.

Closure evidence: `cargo fmt -p hephaestus-wgpu --check`; warning-denied
all-target Clippy; full package Nextest (154/154, 97.521 s); exact final focused
Nextest (5/5, 6.895 s); runnable doctest (1/1); warning-denied Rustdoc; example
output `dot=70 l2=5.4772253`; and the isolated Criterion run all pass.
`cargo semver-checks check-release -p hephaestus-wgpu --baseline-rev
origin/master` cannot construct its temporary unlocked dependency graph because
Aequitas requires Eunomia `^0.6.0` while Eunomia's Git head is 0.7.0. The
delivered public diff is additive (`PreparedDot`, `PreparedL2Norm`, and their
two constructors); no existing export is removed or changed.
PR #60 merged as `ff7e77536e7d80b09bba1b88b8c23f85238da608`.

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

## HEPH-LAPLACIAN-CONTRACT-1 [arch]

- [x] Replace the local boundary enum with Leto's canonical contract.
- [x] Derive POD spacing coefficients through `Laplacian2D` and expose polarity.
- [x] Delete the local CPU reference stencil from WGPU differential tests.
- [x] Pass focused format, check, Clippy, Nextest (152/152), doctest, and
      warning-denied rustdoc gates.

## HEPH-CUDA-FEATURE-HYGIENE [patch]

- [x] Reproduce decomposition-only dead-code warnings with the CUDA-only
      feature combination.
- [x] Gate pinned host staging and decomposition pipeline keys on both
      `cuda` and `decomposition`.
- [x] Pass warning-denied all-target Clippy for `cuda` and
      `cuda,decomposition`.
- [x] Pass configured Nextest: 109/109.

## HEPH-EUNOMIA-0.6-REFRESH [patch]

- [x] Reproduce the stale-provider failure against Eunomia 0.6.0.
- [x] Advance Eunomia, Hermes, and Leto to their merged native-provider
      defaults using Cargo's lock resolver.
- [x] Pass formatting, all-target/all-feature check, warning-denied Clippy,
      configured Nextest, doctests, and warning-denied rustdoc.
- [x] Merge Hephaestus PR #51 at `594d57a`; Atlas PR #44 records that default
      and ATLAS-INTEGRATION-028 tracks this PM-only closeout commit.

**Evidence:** the dependency lock resolves Eunomia `df77dfde`, Hermes
`c9bbdf8a`, and Leto `7afcbd0e`; the full compile and documentation gates pass,
and configured Nextest is 312/312 across CPU, CUDA, WGPU, Metal, and Python
contracts.

## HEPH-EUNOMIA-0.4-REFRESH [patch]

- [x] Advance the reproducibility lock from Eunomia 0.2.0 `34d0cc8a` to
      Eunomia 0.4.0 `49dc115e`.
- [x] Confirm WGPU, CUDA, Metal, and Python compile against the merged provider.
- [x] Pass formatting, warning-denied all-target/all-feature Clippy, configured
      Nextest, doctest, and warning-denied rustdoc gates.

**Evidence:** the dependency lock resolves one Eunomia 0.4.0 source identity;
the complete compile and documentation gates pass, and configured Nextest is
312/312.

## HEPH-EUNOMIA-COMPLEX-1 [arch]

- [x] Audit production, test, and transitive complex-provider ownership.
- [x] Merge Eunomia 0.2.0 and record the public type migration in ADR 0010.
- [x] Replace WGPU, CUDA, Metal, and Python complex buffer types with
      `eunomia::Complex`.
- [x] Remove the Python `eunomia::Complex32` to `numpy::Complex32` conversion
      allocation by using Eunomia's NumPy 0.29 element contract directly.
- [x] Remove direct `num-complex` manifest ownership and commit the updated
      workspace lock.
- [x] Pass format, all-target checks, warning-denied Clippy, Nextest, doctests,
      rustdoc, residue, and SemVer gates.
- [x] Publish and merge the 0.17.0 consumer cutover as PR #48
      (`82bb3a7`).

**Evidence:** affected all-target/all-feature checks and warning-denied Clippy
pass; supported minimal decomposition feature combinations compile; WGPU,
CUDA, Metal, and Python Nextest pass 264/264; doctests and warning-denied
rustdoc pass; public API SemVer analysis reports no additional incompatible
surface; direct source/manifest residue is zero; and the installed PyO3 module's
general eigenvalue result passes the targeted NumPy parity regression.

## HEPH-LEGACY-MATH-RESIDUE-1 [patch]

- [x] Remove the obsolete `ndarray`/`nalgebra` manifest edges and migrate
      WGPU differential oracles to Leto-owned implementations.
- [x] Keep comparative benchmarks real and provider-focused by measuring
      Leto CPU operations against WGPU/CUDA, with no legacy baseline crates.
- [x] Run formatting, locked checks, warning-denied diagnostics, Nextest,
      doctests, rustdoc, and the source-residue audit.

**Evidence:** core 48/48, WGPU 140/140, CUDA 109/109; all-target checks and
warning-denied Clippy pass with the MinGW LLVM path required by the provider's
documented Windows build contract. `numpy` remains only in the Python FFI
boundary, not in provider computation or test/benchmark references.

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

## HEPH-SCAN-TILED-1 order-preserving tiled scan [minor]

- [x] Claim the provider-owned scan slice and record its exact backend/core
  scope (Codex, `codex/hephaestus-tiled-scan`).
- [x] Change the core dispatch contract to one workgroup/block per scan line.
- [x] Generate shared-memory tiled kernels for WGPU and CUDA; retain logical
  chunk order for every scalar type and direction, with floating-point
  reassociation documented and bounded.
- [x] Add theorem/spec Rustdoc and value-semantic core/backend contract coverage.
- [x] Run focused formatting, checks, warning-denied Clippy, nextest, and
  bench compilation. Core 48/48, WGPU 140/140, and CUDA 109/109; doctests and
  rustdoc pass for all three packages.

Evidence: ADR 0009; WGPU and CUDA long-line real-device contracts; source
contract tests; `cargo clippy -p hephaestus-core -p hephaestus-wgpu
-p hephaestus-cuda --all-targets --no-deps -- -D warnings`; and the matching
CUDA no-default-features Clippy run. HEPH-CUDA-CONCURRENT-1 closes the former
Windows access violation.

## HEPH-CUDA-CONCURRENT-1 driver initialization [patch]

- [x] Claim the concurrent-acquisition residual and restrict the scope to
  `hephaestus-cuda/src/infrastructure/device.rs`, its concurrency contract,
  and synchronized PM records (Codex,
  `codex/hephaestus-cuda-init-serialization`).
- [x] Make provider driver initialization single-assignment and thread-safe;
  preserve the typed unavailable-driver error.
- [x] Re-run the concurrent real-device contract, package gates, and docs;
  close the environment residual after the exact abort is gone. The full CUDA
  package nextest passes 109/109.

## Typed WGPU downlevel limits [minor]

- [x] Add `WgpuDevice::downlevel_device_limits` as the provider-owned typed
  mapping of WGPU's downlevel baseline, without exposing WGPU limits to
  consumers.
- [x] Add a value-semantic mapping regression and verify the WGPU package
  diagnostics, nextest, doctest, rustdoc, and minor SemVer classification.
- [x] Merge the 0.16.0 provider release increment as PR #40.

## Typed downlevel acquisition contract [patch]

- [x] Reproduce the conversion defect: unmapped WGPU limits were rebuilt from
  ordinary defaults, so a typed downlevel request became incompatible.
- [x] Preserve the full WGPU downlevel baseline during typed acquisition and
  prove it with an exact mapping regression plus complete WGPU package gates.
- [x] Merge patch release PR #41, update CFDrs to merged 0.16.1, and prove
  typed acquisition on cfd-core GPU nextest 245/245, cfd-math GPU 362/362, and
  cfd-2d GPU 570/570 (27 pre-existing skips) in merged CFDrs PR #295.

## Odd-length WGPU storage padding [patch]

- [x] Record ADR 0008: logical scalar length remains authoritative while WGPU
  owns physical four-byte padding at allocation and transfer boundaries.
- [x] Remove the core-wide four-byte logical-length rejection without removing
  overflow validation.
- [x] Pad only the WGPU storage/upload/full-write byte payload and retain the
  original typed buffer length plus exact host readback.
- [x] Add core and real-device value-semantic regressions for a 27-element
  `u16` payload; pass focused diagnostics, Nextest, doctest, and rustdoc.
- [x] Update Apollo to the merged provider revision and pass its native-f16
  consumer integration gate (Apollo merge commit `26f433e3`).

## CUDA Bindgen toolchain selection [patch]

- [x] Reproduce the CUDA bindgen loader failure against the UCRT LLVM
  distribution and verify that the installed MinGW LLVM distribution loads.
- [x] Set `LIBCLANG_PATH` and prepend the MinGW LLVM directory to `PATH`, then
  pass `hephaestus-cuda --all-targets --locked` plus the formerly blocked
  core/WGPU all-target, all-feature check. This is compile-time evidence only;
  CUDA execution remains a separate device contract.

## Provider default-source convergence [minor]

- [x] Advance the root to the current default branch and remove every Leto,
  Mnemosyne, Moirai, and Themis revision requirement from the workspace SSOT.
- [x] Publish Rust 1.95 through the workspace package metadata and every
  member manifest; record the pre-1.0 0.15.0 release decision in ADR 0007.
- [x] Regenerate the ignored library lockfile and verify one identity for each
  provider: Leto `87c67f0`, Mnemosyne `cb103a5`, Moirai `4ad6520`, and Themis
  `709aec6`. Rust 1.95 checks `hephaestus-wgpu`; Rust 1.94.1 rejects the
  resolved graph, including the declared Hephaestus packages.
- [x] Pass formatting, warning-denied release WGPU Clippy, release nextest,
  doctests, rustdoc, and 196/196 applicable minor semver checks before
  publishing 0.15.0.
- [x] Update Apollo's direct provider graph after this Hephaestus source change
  merges; Apollo merge commit `26f433e3` locks one provider identity in its
  transform consumer packages.

## Required device-feature acquisition [minor]

- [x] Add a `DeviceFeature`-typed required acquisition entry point to
  `WgpuDevice`, preserving backend selection while rejecting adapters that
  cannot enable every requested feature.
- [x] Verify the feature-set mapping, focused WGPU check, warning-denied
  Clippy, 133-case WGPU nextest, doctest, rustdoc, 196/196 applicable minor
  semver checks, and release records. Apollo's consumer pin is the next
  dependency-ordered increment.

## Prefix-copy provider contract [minor]

- [x] Add the length-checked `CommandStream::copy_prefix` contract and native
  WGPU/CUDA implementations.
- [x] Prove WGPU prefix-copy ordering and suffix preservation with a real-device
  value-semantic nextest regression.

## WGPU 30 provider ABI [major]

- [x] Verify current registry WGPU is 30.0.0 with Rust 1.87 MSRV and record the
  provider-first migration decision in ADR 0006.
- [x] Update the workspace WGPU SSOT and all native Hephaestus call sites until
  warning-denied WGPU/Python/Metal compilation passes.
- [x] Run formatting, warning-denied Clippy, nextest, doctest, rustdoc, RustSec,
  dependency policy, and semver classification gates.
- [x] Bump the workspace to 0.13.0, synchronize changelog/backlog/gap audit,
  commit, and push the provider release increment.
- [x] Repin Apollo to the pushed provider commit and repeat its complete
  release gate before advancing Atlas gitlinks (Apollo merge commit
  `26f433e3`).

## Themis provider identity [patch]

- [x] Replace the obsolete Themis 0.6 revision with the exact current Git
  revision used by Hermes and Leto.
- [x] Remove the root-only path patch that downstream consumers could not see.
- [x] Pass focused Hephaestus WGPU gates and confirm the provider graph resolves
  one Themis 0.10 identity. Evidence: warning-denied `hephaestus-wgpu` clippy,
  132 focused nextest cases, and inverse dependency inspection all pass.
- [x] The previous revision quarantine is deleted by HEPH-PROVIDER-DEFAULT-2;
  no compatibility source policy remains on the release branch.

## Verified locally — HEPH-EMPTY-001 genuine empty decompositions [patch]

- [x] Enumerate every synthetic 1x1 branch and verify Leto's canonical
  decompositions already represent the actual empty dimensions.
- [x] Add CUDA/WGPU value-semantic regressions for empty identity factors,
  determinant, rank, permutations, and shapes.
- [x] Delete the synthetic branches and route empty inputs through the same
  canonical Leto representation as nonempty inputs.
- [x] Pass formatting, focused CUDA/WGPU contracts, Clippy, all 239 package
  tests, doctests, and warning-clean package documentation.
- [x] Synchronize the release artifacts for the stacked Hephaestus commit and
  Atlas integration-pointer advance.

## Superseded — WGPU-CB-1 immutable staging callbacks [major]

- [x] Reconcile the stale broad KS-3 claim and reclaim only `device.rs` plus
  its contract test.
- [x] Make `WgpuDevice::new` return typed `Result` and register one static
  Mnemosyne callback pair before publishing the staging device.
- [x] Update all package-local constructors and contract tests.
- [x] Run focused and full `hephaestus-wgpu` gates: check and clippy pass;
  nextest passes 131/131; doctests and rustdoc pass.
- [x] Commit and push the consumer change.
- [ ] Semver gate: make the baseline clone resolve repository-external sibling
  path dependencies. The current 0.12.0 rustdoc build passes, but the baseline
  clone cannot find `../leto/crates/leto`; the local Atlas graph is green with
  Moirai's committed Mnemosyne 0.2 requirement and no consumer-tree edits.

2026-07-06 (KS-8 WDDM launch-drain recheck). Verified the CUDA launch SSOT in
`crates/hephaestus-cuda/src/application/pipeline.rs` carries the Windows-gated
post-launch `cuCtxSynchronize` drain after `cuLaunchKernel`, and updated the
launch Rustdoc so Windows completion/error behavior is no longer described as
asynchronous. Evidence tier: value-semantic live-CUDA nextest. Checks: `cargo
nextest run -p hephaestus-cuda reduction_sum_matches_cpu_reference
reduction_min_max_matches_cpu_reference reduction_width_is_part_of_dispatch_contract
reduction_axis_reduction_generic_matches_cpu linalg_dot_matches_cpu_reference
linalg_trace_matches_cpu_reference linalg_norms_match_cpu_reference
hessenberg_reconstructs_and_preserves_similarity_invariants
non_default_block_width_produces_identical_results` passes 9/9, and `cargo
nextest run -p hephaestus-cuda concurrent_device_acquisition_is_safe` passes
1/1. Residual tracking is limited to the documented concurrent-device-acquisition
case; current local evidence shows it passing rather than aborting.

2026-07-05 (KS-5 reduction parity increment). Delivered this session:
`hephaestus_core::reduction` is now the SSOT for CUDA/WGPU axis-reduction
planning and scalar reduction host planning. `validate_reduction_width` and
`reduction_pass_count` moved out of both backend reduction modules, and both
backends import the core helpers while retaining only shader/source, buffer,
and launch ownership. CUDA default reduction checks required a same-contract
`leto::Complex<f32>` -> `num_complex::Complex<f32>` eigenvalue upload fix; the
comparative benchmark now uses one local conversion helper for the same Leto
complex type drift. Evidence tier: compile-time validation, clippy, and
value-semantic nextest. Checks: `cargo fmt -p hephaestus-core -p
hephaestus-wgpu -p hephaestus-cuda --check`, `cargo check -p hephaestus-core`,
`cargo check -p hephaestus-cuda --no-default-features`, `cargo check -p
hephaestus-wgpu`, `cargo check -p hephaestus-cuda`, `cargo nextest run -p
hephaestus-core reduction` (6/6), `cargo nextest run -p hephaestus-cuda
--no-default-features reduction` (4/4), `cargo nextest run -p hephaestus-cuda
reduction` (4/4), `cargo nextest run -p hephaestus-wgpu reduction` (5/5), and
`cargo clippy -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda
--all-targets --no-deps -- -D warnings`.

Next increment: [KS-5] continue host-orchestration consolidation with the
blocked-decomposition host-loop planners, then wrapper match-arm collapse where
the core planner contracts make backend branching redundant.

2026-07-02 (ADR-0004 kernel-seam programme, claude-seam session, branch
arch/kernel-seam). Delivered this session: [KS-1] core dialect + op
vocabulary; [KS-2] authored-kernel seam (KernelInterface/KernelSource/
KernelDevice/CommandStream/Binding); [KS-3] both backends consume the core
vocabulary (per-backend trait pairs and ZSTs deleted, CUDA templates on
canonical lhs/rhs, net −800 lines); [KS-4] KernelDevice/CommandStream impls
both backends incl. grouped variant (concurrent kwavers stream) + device
capabilities; [KS-6] python monolith split into 12 leaf modules,
split_packed_lu evicted to core; CUDA correctness batch (bind-per-launch
SSOT, success-only compile cache, NVRTC codes, context-aware unload, honest
stub launches); wgpu safety/memory batch (HostPinned staging-device gate,
pool budgets, uniform leak, SAFETY pass); O(N) axis scan both backends
(2.65x bench); wgpu dot/norm_l2/norm_max fusion; CUDA SAFETY closure;
version 0.11.0 + CHANGELOG with Breaking/Migration. External acceptance:
helios `GpuAttenuationMapper` authored the H-010b HU→μ fused affine-clamp kernel
over the seam with zero type-specific substrate helper. Evidence tier:
differential/empirical validation: `rustup run nightly cargo nextest run -p
helios-gpu attenuation` passes 5/5 in atlas `repos/helios`, and `rustup run
nightly cargo nextest run -p hephaestus-core -p hephaestus-wgpu stream` passes
8/8 for the supporting WGSL authored-kernel seam.

Previous next increment: [KS-5] hoist per-family host orchestration into core
generic over the seam; scan and reduction planner parity are now delivered,
with blocked-decomposition host loops and wrappers still remaining.

2026-07-05 (CUDA Stage 1 substrate reconciliation). Replaced the Stage 1
cutile/Mnemosyne managed-memory substrate with ADR-0001's cuda-oxide-owned
driver initialization, context creation/binding, `cuMemAlloc_v2` allocation,
checked `cuMemcpy*` upload/download/subrange transfer, and context-bound
`cuMemFree_v2` release. `CudaBuffer<T>` remains typed by `PhantomData<T>` and
retains its cuda-oxide context so frees and module unloads bind the owning
context before driver calls. The CUDA placement resolver records every
allocatable primary-buffer hint as non-managed `MemoryTier::Device`, rejects
budget-only tiers, and reports `MappablePrimaryBuffers` as unsupported because
host access is explicit copy-only. The blocked-decomposition region helper uses
row-wise 1D copies because cuda-oxide 0.4.0's `CUDA_MEMCPY2D` layout is not
ABI-correct on Windows/MSVC. Evidence tier: compile-time validation, clippy,
rustdoc, and value-semantic live-CUDA/no-driver contract tests. Checks: `cargo fmt -p
hephaestus-cuda --check`, `cargo check -p hephaestus-cuda`, `cargo check -p
hephaestus-cuda --no-default-features`, `cargo clippy -p hephaestus-cuda
--all-targets --no-deps -- -D warnings`, `cargo clippy -p hephaestus-cuda
--no-default-features --all-targets --no-deps -- -D warnings`, `cargo nextest
run -p hephaestus-cuda` passes 105/105 on live CUDA, `cargo nextest run -p
hephaestus-cuda --no-default-features` passes 60/60 via skip-without-driver
contracts, `cargo test --doc -p hephaestus-cuda` passes 0 doctests, and `cargo
doc -p hephaestus-cuda --no-deps` passes. Residual tracking is limited to the
documented concurrent-device-acquisition case, rechecked in the 2026-07-06
KS-8 WDDM launch-drain entry above. Build note: `cuda-oxide` 0.4.0's build
script links `cuda.lib`, so this repository sets `CUDA_LIB_PATH` for the
default CUDA feature even though no-default stub verification still compiles
without a CUDA driver/device.

2026-07-03 (CUDA ComputeDeviceCapabilities). Implemented
`ComputeDeviceCapabilities` for `CudaDevice` with driver-backed limits and
explicit backend semantics. Block dimensions, threads per block, shared memory
per block, compute capability, host-mapping support, unified addressing, and
current memory are read through CUDA driver APIs. `DeviceLimits` now represents
per-shader-stage storage-buffer slots as `Option<u32>`: WGPU reports `Some`,
CUDA reports `None` because authored CUDA kernels receive flat arguments. The
no-CUDA stub is uninhabited for capability queries, so stub mode compiles
without returning fabricated limits. Evidence tier: compile-time validation,
clippy, focused value-semantic nextest, and downstream integration checks.
Checks: `cargo fmt -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda
--check`, `cargo check -p hephaestus-core`, `cargo check -p hephaestus-wgpu`,
`cargo check -p hephaestus-cuda`, `cargo check -p hephaestus-cuda
--no-default-features`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu -p
hephaestus-cuda --all-targets --no-deps -- -D warnings`, `cargo clippy -p
hephaestus-cuda --no-default-features --all-targets --no-deps -- -D warnings`,
`cargo nextest run -p hephaestus-cuda device_capabilities_are_driver_backed`
passes 1/1, `cargo nextest run -p hephaestus-cuda --no-default-features
device_capabilities_are_driver_backed` passes 1/1, downstream `cargo check -p
kwavers-gpu --features gpu`, `cargo clippy -p kwavers-gpu --features gpu
--all-targets --no-deps -- -D warnings`, `cargo nextest run -p kwavers-gpu
--features gpu backend` passes 31/31, `cargo nextest run -p kwavers-gpu
--features gpu multi_gpu` passes 3/3, and `cargo nextest run -p kwavers
--features gpu --test gpu_device_tests` passes 9/9.

2026-07-02 (ComputeDevice capability trait). Added
`ComputeDeviceCapabilities` as the backend-neutral trait seam for provider
feature checks and enabled device limits. WGPU implements the trait and adds a
`DeviceFeature`/`DeviceLimits` constructor so downstream contexts can request
capabilities without building WGPU descriptors. Driver: Kwavers `WGPUContext`
and `CoreGpuContext` now store a generic `D: ComputeDeviceCapabilities`, with
WGPU only as the default acquisition specialization. Evidence tier:
compile-time validation, clippy, focused downstream value-semantic nextest, and
source audit. Checks: `cargo fmt -p hephaestus-core -p hephaestus-wgpu
--check`, `cargo check -p hephaestus-core`, `cargo check -p hephaestus-wgpu`,
`cargo clippy -p hephaestus-core -p hephaestus-wgpu --all-targets --no-deps --
-D warnings`, downstream `cargo fmt -p kwavers-gpu --check`, `cargo check -p
kwavers-gpu --features gpu`, `cargo clippy -p kwavers-gpu --features gpu
--all-targets --no-deps -- -D warnings`, `cargo nextest run -p kwavers-gpu
--features gpu backend` passes 31/31, and `cargo nextest run -p kwavers-gpu
--features gpu multi_gpu` passes 3/3. Follow-up on 2026-07-03 implemented the
CUDA capability trait with driver-backed limits and `None` for WGPU-only
storage-binding slots.

2026-07-02 (WGPU provider capability accessors). Added provider-owned
`WgpuDevice::features()` and `WgpuDevice::limits()` so downstream crates can
query capability metadata without borrowing raw WGPU device handles. Driver:
Kwavers backend contexts removed public raw `wgpu::Device`/`wgpu::Queue`
accessors while preserving capability reporting, synchronization, and typed
transfer behavior. Evidence tier: compile-time validation, clippy, focused
downstream value-semantic nextest, and source audit. Checks: `cargo fmt -p
hephaestus-wgpu --check`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
hephaestus-wgpu --all-targets --no-deps -- -D warnings`, downstream `cargo
check -p kwavers-gpu --features gpu`, `cargo clippy -p kwavers-gpu --features
gpu --all-targets --no-deps -- -D warnings`, and `cargo nextest run -p
kwavers-gpu --features gpu backend device multi_gpu` passes 34/34. Residual:
Hephaestus still exposes raw handles for backend internals and migration-only
consumers; new downstream capability reporting should use these provider
accessors.

2026-07-02 (ComputeDevice sub-buffer transfer seam). Added
`ComputeDevice::write_sub_buffer` so consumers can refresh typed device-buffer
subranges through the backend-neutral trait instead of concrete WGPU/CUDA
handles. WGPU and CUDA delegate to their checked concrete subrange transfer
paths, Metal delegates to its wrapped WGPU provider, and the CUDA-unavailable
stub returns the existing typed unavailable error. Driver: Kwavers PSTD
source/velocity run-cache signal tails now call the provider trait and remain
substitutable by CUDA/Metal provider buffers when those kernels are authored.
Evidence tier: compile-time validation, clippy, and value-semantic backend
nextest. Checks: `cargo fmt -p hephaestus-core -p hephaestus-wgpu -p
hephaestus-cuda -p hephaestus-metal --check`, `cargo check -p hephaestus-core
-p hephaestus-wgpu -p hephaestus-cuda -p hephaestus-metal --all-targets
--no-default-features`, `cargo check -p hephaestus-cuda`, `cargo clippy -p
hephaestus-cuda --all-targets --no-deps -- -D warnings`, `cargo clippy -p
hephaestus-core -p hephaestus-wgpu -p
hephaestus-cuda -p hephaestus-metal --all-targets --no-default-features
--no-deps -- -D warnings`, and `cargo nextest run -p hephaestus-wgpu -p
hephaestus-cuda -p hephaestus-metal --no-default-features write_sub_buffer`
passes 9/9. Residual: consumer command encoders and shader bindings still need
their own grouped-kernel migration where they remain WGPU-specific.

2026-07-02 (Grouped authored-kernel provider seam). Added the
backend-neutral grouped authored-kernel API for consumer kernels that require
multiple WGPU bind groups while staying CUDA-flat. `GroupedKernelInterface` /
`GroupedKernelSource<L>` declare grouped storage bindings, parameter
group/binding, and launch shape; `GroupedKernelDevice` /
`GroupedCommandStream` prepare and encode grouped kernels. WGPU now builds one
bind group per declared group and injects the POD parameter block as a uniform
at the declared group/binding; CUDA validates the same grouped contract and
launches storage buffers as device-pointer arguments in declaration order plus
the POD parameter block by value. Driver: Kwavers PSTD can now move its
multi-group field/kspace/sensor/absorption kernels onto a Hephaestus WGPU/CUDA
provider trait instead of a Kwavers raw-WGPU command helper. Evidence tier:
compile-time validation, clippy, and value-semantic WGPU/CUDA nextest. Checks:
`cargo fmt -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda --check`,
`cargo check -p hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo check
-p hephaestus-cuda --no-default-features`, `cargo clippy -p hephaestus-core -p
hephaestus-wgpu -p hephaestus-cuda --all-targets --no-deps -- -D warnings`,
`cargo nextest run -p hephaestus-wgpu grouped_command_stream` passes 2/2, and
`cargo nextest run -p hephaestus-cuda --no-default-features
cuda_grouped_command_stream` passes 2/2. Residual: consumer kernels such as
Kwavers PSTD still need their WGSL parameter ABI migrated from push constants
to the grouped uniform contract and corresponding CUDA C sources authored.

2026-07-02 (Grouped same-region sequence seam). Added
`GroupedKernelSequence` and `GroupedCommandStream::encode_grouped_sequence`
for ordered grouped dispatches that must stay inside one backend-defined
region. WGPU implements the sequence as one compute pass; CUDA implements it
as ordered launches on the bound CUDA stream. Driver: Kwavers PSTD can migrate
FFT/source/density/pressure/record/absorption timestep kernels without
splitting its current same-pass WGPU dispatch order or adding a downstream
helper. Evidence tier: compile-time validation, clippy, and value-semantic
WGPU/CUDA nextest. Checks: `cargo fmt -p hephaestus-core -p hephaestus-wgpu
-p hephaestus-cuda --check`, `cargo check -p hephaestus-core`, `cargo check
-p hephaestus-wgpu`, `cargo check -p hephaestus-cuda --no-default-features`,
`cargo check -p hephaestus-cuda`, `cargo clippy -p hephaestus-core -p
hephaestus-cuda --all-targets --no-default-features --no-deps --
-D warnings`, `cargo clippy -p hephaestus-wgpu --all-targets --no-deps --
-D warnings`, `cargo nextest run -p hephaestus-wgpu stream` passes 8/8, and
`cargo nextest run -p hephaestus-cuda --no-default-features stream` passes
6/6. Residual: each consumer grouped timestep kernel still needs a
`GroupedKernelSource<CudaC>` implementation before it can execute on CUDA.

2026-07-02 (CUDA authored-kernel command stream). Implemented
`KernelDevice`/`CommandStream` for `CudaDevice`, closing KS-4 for both WGPU and
CUDA provider backends. The CUDA implementation prepares NVRTC-compiled
`KernelSource<CudaC>` kernels from the shared `KernelInterface` contract,
launches typed storage bindings as device-pointer arguments, passes the POD
parameter block by value, and uses CUDA default-stream ordering for
dispatch/copy/fill. `hephaestus-cuda --no-default-features` now compiles
honestly by making `leto-ops` a real dependency of modules that already use it,
marking the comparative bench as `decomposition`-required, and gating
decomposition-only test helpers. Evidence tier: compile-time validation,
clippy, and value-semantic CUDA nextest. Checks: `cargo fmt -p hephaestus-cuda
--check`, `cargo check -p hephaestus-cuda`, `cargo check -p hephaestus-cuda
--no-default-features`, `cargo clippy -p hephaestus-cuda --all-targets
--no-deps -- -D warnings`, `cargo clippy -p hephaestus-cuda
--no-default-features --all-targets --no-deps -- -D warnings`, `cargo nextest
run -p hephaestus-cuda stream` passes 3/3, and `cargo nextest run -p
hephaestus-cuda --no-default-features stream` passes 3/3. Residual: concrete
consumer CUDA kernels still need to be authored where consumers only have WGSL
sources.

2026-07-02 (WGPU authored-kernel command stream). Implemented
`KernelDevice`/`CommandStream` for `WgpuDevice`, giving downstream consumers a
backend-neutral authored-kernel seam for WGSL `KernelSource<Wgsl>` kernels.
The WGPU implementation prepares pipelines from the shared `KernelInterface`
binding contract, records ordered dispatch/copy/zero-fill passes, validates
typed storage bindings, submits through the provider stream boundary, and
reuses the existing uniform-buffer pool. Evidence tier: compile-time validation,
clippy, and value-semantic WGPU nextest. Checks: `cargo fmt -p
hephaestus-wgpu --check`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
hephaestus-wgpu --all-targets --no-deps -- -D warnings`, and `cargo nextest run
-p hephaestus-wgpu stream` pass 5/5. CUDA now implements the same seam through
`KernelSource<CudaC>`.

2026-07-02 (WGPU dialect trait migration repair). Completed the remaining
`hephaestus-wgpu` migration away from deleted backend-local shader traits.
Linalg, random, sparse, scan exports, and crate exports now consume shared
`hephaestus_core` dialect traits (`DialectScalar`, `UnaryExpr`, `BinaryExpr`,
`CombineExpr`, `IdentityToken`, `OpIdentity`) instead of stale
`WgslScalar`/operation-specific aliases. Evidence tier: compile-time
validation plus static source audit. Checks: stale-name source audit clean,
`cargo check -p hephaestus-wgpu`, and `cargo clippy -p hephaestus-wgpu
--all-targets --no-deps -- -D warnings` pass. Driver: Kwavers `GpuDevice`
provider-surface migration compiles and passes focused GPU device nextest.

2026-07-02 (WGPU storage-kernel lint cleanup). Removed a stale
`DeviceExt` import from `hephaestus-wgpu` storage-kernel dispatch so downstream
provider builds no longer carry the unused-import warning. Evidence tier:
compile-time validation. Check: `cargo check -p hephaestus-wgpu` passes.

2026-07-02 (ComputeDevice synchronization seam). Added
`ComputeDevice::synchronize` so downstream GPU consumers can request explicit
blocking semantics without depending on WGPU polling or CUDA context calls.
`hephaestus-wgpu` maps it to `Device::poll`, `hephaestus-cuda` maps it to
`cuCtxSynchronize`, `hephaestus-metal` delegates to the wrapped WGPU device, and
the CUDA-unavailable stub returns the existing typed unavailable error. Driver:
Kwavers visualization `DataPipeline<D>` is now generic over `ComputeDevice` and
uses provider buffers plus `write_buffer`/`synchronize` instead of raw WGPU
device/queue ownership. Evidence tier: compile-time validation plus downstream
value-semantic nextest. Checks: `cargo check -p hephaestus-core -p
hephaestus-wgpu -p hephaestus-metal`, `cargo check -p hephaestus-cuda`, `cargo
fmt --all --check`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu -p
hephaestus-cuda -p hephaestus-metal --all-targets -- -D warnings`, `cargo
nextest run -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda -p
hephaestus-metal storage_kernel` passes 2/2, and downstream
`cargo nextest run -p kwavers-analysis --features gpu-visualization
visualization` passes 15/15.

2026-07-02 (Backend-neutral multi-storage kernel dispatch). Added
`hephaestus_core::MultiStorageKernel<D, P, B>` for kernels whose storage layout
is wider than unary/binary. `hephaestus-wgpu` now provides
`WgslMultiStorageKernel`, `WgslStorageBindingLayout`, and `WgslStorageBinding`,
owning the real WGPU shader module, bind-group layout, uniform-buffer pool
usage, bind group, encoder, and workgroup submission for N storage buffers plus
one POD parameter block. Downstream Kwavers 3-D static DAS and dynamic-focus
DAS now consume this provider path instead of local bind-group/compute-pass
construction. Evidence tier: compile-time validation plus value-semantic
layout validation and downstream beamforming nextest. Checks: `cargo check -p
hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
hephaestus-core -p hephaestus-wgpu --all-targets -- -D warnings`, `cargo
nextest run -p hephaestus-core -p hephaestus-wgpu storage_kernel` passes 2/2,
`cargo check -p kwavers-analysis --features gpu`, `cargo clippy -p
kwavers-analysis --features gpu --all-targets --no-deps -- -D warnings`, and
`cargo nextest run -p kwavers-analysis --features gpu three_dimensional` passes
52/52. Residual: CUDA needs a concrete multi-storage beamforming kernel
implementor when the CUDA kernel exists.

2026-07-03 (Multi-storage binding constructor). Added
`hephaestus_core::MultiStorageDevice`, giving generic consumers a backend-owned
`storage_binding(binding, &D::Buffer<T>)` constructor for the binding bundle
used by `MultiStorageKernel`. `WgpuDevice` implements it with
`WgslStorageBinding`, so downstream structs can stay generic over the provider
while WGPU keeps its concrete bind-group representation. Evidence tier:
type-level provider boundary plus compile-time validation, clippy, downstream
value-semantic nextest, and source audit. Checks: `cargo fmt -p
hephaestus-core -p hephaestus-wgpu --check`, `cargo check -p hephaestus-core -p
hephaestus-wgpu`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu
--all-targets --no-deps -- -D warnings`, `cargo nextest run -p hephaestus-core
-p hephaestus-wgpu storage_kernel` passes 2/2, downstream `cargo check -p
kwavers-analysis --features gpu`, `cargo clippy -p kwavers-analysis --features
gpu --all-targets --no-deps -- -D warnings`, and `cargo nextest run -p
kwavers-analysis --features gpu three_dimensional` pass 52/52.

2026-07-02 (Backend-neutral storage kernel dispatch). Added
`hephaestus_core::DispatchGrid`, `UnaryStorageKernel<D, T, P>`, and
`BinaryStorageKernel<D, T, P>` so downstream crates can dispatch one-input and
two-input storage-buffer kernels generically over a `ComputeDevice`
implementor. `hephaestus-wgpu` now provides `WgslUnaryStorageKernel` and
`WgslBinaryStorageKernel`, which own the real WGPU shader module, pipeline,
uniform buffer, bind groups, encoder, and workgroup submission for the current
storage-kernel layouts. Evidence tier: compile-time validation plus
value-semantic launch-grid tests. Checks: `cargo fmt -p hephaestus-core -p
hephaestus-wgpu`, `cargo check -p hephaestus-core`, `cargo check -p
hephaestus-wgpu`, and `cargo nextest run -p hephaestus-core kernel` passes 2/2.

2026-07-01 (WGPU tiled axis-0 reductions). Added a WGPU axis-0 tiled reduction
kernel for rank-2 reductions so one workgroup reduces up to 16 output columns
instead of launching one workgroup per output element. This preserves the
existing generic axis API and fallback paths while reducing the 256x256 axis-0
case from 256 workgroups to 16. Evidence tier: value-semantic contract tests and
empirical benchmark: `cargo nextest run -p hephaestus-wgpu
reduction_sum_matches_cpu_reference axis_reductions_match_leto_reference` (2/2);
`HEPHAESTUS_BENCH_DISABLE_CUDA=1 cargo bench -p hephaestus-wgpu --bench
comparative` (prepared final-pass scalar sum: WGPU 42.702 µs, Leto 63.090 µs,
ndarray 85.468 µs; axis sum: WGPU 22.136 µs, Leto 10.446 µs, ndarray 6.528 µs;
axis min: WGPU 20.726 µs, Leto 5.406 µs, ndarray 4.634 µs; axis max: WGPU
11.778 µs, Leto 5.360 µs, ndarray 4.422 µs; axis mean: WGPU 18.048 µs, Leto
7.172 µs, ndarray 5.876 µs). Residual parity gap: WGPU scalar sum beats ndarray,
and the tiled axis kernel removes the previous max/mean outliers, but CPU still
wins the 256x256 axis-0 shape. The next reduction slice should add measured
small-axis routing or fuse multiple axis statistics into one GPU pass.

2026-07-01 (Cross-layer reduction rerun with Leto axis fast path). Added a
Leto `leto-ops` row-major rank-2 axis-0 fast path for contiguous CPU axis
reductions, then reran the Hephaestus comparative harness against the local Leto
provider. Evidence tier: value-semantic Leto axis tests plus empirical
Hephaestus comparative benchmark: `cargo nextest run -p leto-ops reduction
sum_mean_axis_match_ndarray` (16/16). The Leto CPU axis fast path is now at or
near ndarray for the 256x256 axis-0 shape.

2026-07-01 (WGPU final-pass and batched axis reductions). Added a final scalar
reduction WGSL kernel that lets one workgroup fold up to `BlockWidth *
BlockWidth` partials, reducing the $2^{20}$ scalar sum tree from three compute
passes to two. Added `submit_prepared_reduction_batch` and
`submit_prepared_axis_reduction_batch` over independent prepared outputs; the
comparative axis rows now measure warmed batched prepared axis reductions.
Evidence tier: value-semantic CPU/Leto differential contracts plus empirical
comparative benchmark: `cargo fmt -p hephaestus-wgpu`; `cargo check -p
hephaestus-wgpu --bench comparative`; `cargo nextest run -p hephaestus-wgpu
reduction_sum_matches_cpu_reference axis_reductions_match_leto_reference` (2/2);
`HEPHAESTUS_BENCH_DISABLE_CUDA=1 cargo bench -p hephaestus-wgpu --bench
comparative` (prepared final-pass scalar sum: WGPU 112.584 µs, Leto 61.218 µs,
ndarray 80.860 µs; warmed batched prepared axis sum: WGPU 11.596 µs, Leto
43.756 µs, ndarray 3.802 µs; axis min: WGPU 11.770 µs, Leto 46.962 µs, ndarray
4.452 µs; axis max: WGPU 14.456 µs, Leto 48.192 µs, ndarray 4.412 µs; axis
mean: WGPU 8.826 µs, Leto 42.102 µs, ndarray 4.640 µs). Residual parity gap:
axis reductions now beat Leto but not `ndarray`; scalar sum remains slower than
`ndarray` on the latest full comparative run. Lower-level follow-up should
target WGPU pass/submit planning in Hephaestus/Mnemosyne/Moirai before touching
Hermes SIMD or Leto CPU arithmetic.

2026-07-01 (WGPU prepared scalar reduction). Added `PreparedReduction` plus
`prepare_reduction`/`prepare_reduction_with_width` so repeated scalar
reductions over a fixed input can reuse the compiled pipeline, tree scratch
buffers, and bind groups. The comparative sum-reduction row now times this
prepared path. Evidence tier: value-semantic CPU-reference contract plus
empirical comparative benchmark: `cargo fmt -p hephaestus-wgpu`; `cargo check
-p hephaestus-wgpu --bench comparative`; `cargo nextest run -p hephaestus-wgpu
reduction_sum_matches_cpu_reference axis_reductions_match_leto_reference` (2/2);
`HEPHAESTUS_BENCH_DISABLE_CUDA=1 cargo bench -p hephaestus-wgpu --bench
comparative` (prepared sum reduction: WGPU 123.488 µs, Leto 59.428 µs, ndarray
84.580 µs; prepared axis sum: WGPU 41.190 µs, Leto 42.052 µs, ndarray 4.800
µs; prepared axis min: WGPU 58.470 µs, Leto 47.432 µs, ndarray 4.598 µs;
prepared axis max: WGPU 45.244 µs, Leto 47.148 µs, ndarray 4.676 µs; prepared
axis mean: WGPU 68.860 µs, Leto 43.200 µs, ndarray 5.456 µs). Residual parity
gap: the prepared scalar path removes allocation/setup churn but does not beat
the old scalar comparative row on this run because repeated queued dispatches
reuse the same scratch/output buffers; the next scalar lever is batched
independent-output prepared reductions or a persistent scratch strategy measured
under dispatch-and-wait semantics.

2026-07-01 (WGPU prepared axis-reduction dispatch). Added
`PreparedAxisReduction` and `prepare_{sum,min,max,mean}_axis_into` so repeated
rank-2 axis reductions over fixed input/output buffers reuse the selected
pipeline, metadata uniform, and bind group. The comparative axis-reduction rows
now time the prepared WGPU path, while the one-shot APIs keep their existing
surface and share the same pipeline-selection helper. Evidence tier:
value-semantic Leto differential contract plus empirical comparative benchmark:
`cargo fmt -p hephaestus-wgpu`; `cargo check -p hephaestus-wgpu --bench
comparative`; `cargo nextest run -p hephaestus-wgpu
axis_reductions_match_leto_reference` (1/1); `HEPHAESTUS_BENCH_DISABLE_CUDA=1
cargo bench -p hephaestus-wgpu --bench comparative` (sum reduction: WGPU
104.712 µs, Leto 69.016 µs, ndarray 80.004 µs; prepared axis sum: WGPU 42.302
µs, Leto 42.508 µs, ndarray 5.136 µs; prepared axis min: WGPU 61.540 µs, Leto
46.442 µs, ndarray 4.888 µs; prepared axis max: WGPU 29.742 µs, Leto 46.974 µs,
ndarray 5.134 µs; prepared axis mean: WGPU 41.566 µs, Leto 40.030 µs, ndarray
5.946 µs). Residual parity gap: prepared axis sum is at Leto parity, axis max
beats Leto, and axis mean is near Leto, but all prepared axis rows still trail
`ndarray` because one WGPU submit/poll remains dominant at 256x256. Scalar sum
still trails `ndarray` on the latest run because it allocates multi-pass
intermediate buffers and creates per-pass bind groups per call.

2026-07-01 (WGPU axis-reduction workgroup path). Replaced the rank-2 axis
sum/min/max/mean default path for reduced axes that fit the selected
`BlockWidth` with one workgroup per output element and a workgroup-memory tree
reduction, while preserving the existing one-thread-per-output shader as the
long-axis fallback. The contract test now value-checks the default parallel path
against Leto and forces the fallback with a narrow width. Evidence tier:
value-semantic Leto differential contract plus empirical comparative benchmark:
`cargo fmt -p hephaestus-wgpu`; `cargo check -p hephaestus-wgpu --bench
comparative`; `cargo nextest run -p hephaestus-wgpu
axis_reductions_match_leto_reference` (1/1); `HEPHAESTUS_BENCH_DISABLE_CUDA=1
cargo bench -p hephaestus-wgpu --bench comparative` (sum reduction: WGPU
83.526 µs, Leto 77.748 µs, ndarray 80.048 µs; axis sum: WGPU 77.384 µs, Leto
42.458 µs, ndarray 5.556 µs; axis min: WGPU 66.898 µs, Leto 48.738 µs,
ndarray 4.968 µs; axis max: WGPU 35.524 µs, Leto 46.774 µs, ndarray 4.802 µs;
axis mean: WGPU 73.462 µs, Leto 42.220 µs, ndarray 6.654 µs). Residual parity
gap: scalar sum is near ndarray on this run and axis max beats Leto, but axis
sum/min/mean remain below Leto/ndarray for 256x256 because fixed WGPU
submit/poll and per-call uniform/bind-group setup dominate the workload. Next
concrete increment: add prepared/reused axis-reduction dispatch for fixed
input/output/layouts before attempting multi-pass long-axis parallelization.

2026-07-01 (Blocked QR delayed work-buffer copy). Moved the blocked-QR full
matrix copy out of the pre-loop critical path. The first panel now downloads
from the original input buffer; immediately after that readback, the full input
copy to `work_buf` is queued so it can overlap the first CPU panel
factorization before any panel write/update touches `work_buf`. Queue ordering
preserves correctness for later panels. Evidence tier: value-semantic blocked
QR contract tests plus empirical component benchmark: `cargo fmt -p
hephaestus-wgpu --check`; `cargo check -p hephaestus-wgpu --bench
decomposition_sync`; `cargo nextest run -p hephaestus-wgpu blocked_qr` (4/4);
`cargo bench -p hephaestus-wgpu --bench decomposition_sync` (QR sync floor
213.209 µs, CPU panel lower bound 26.369 µs, timestamp total 7.776 µs, median
192 ns). Residual parity gap: blocked QR is still transfer/synchronization
bound, but the first-panel dependency chain is reduced.

2026-07-01 (Blocked QR Householder resource reuse). Hoisted the blocked-QR
Householder metadata uniform buffer, bind group, and host reflector-metadata
scratch out of the panel loop. Each panel now rewrites the invariant uniform and
reflector buffers but does not recreate the bind group or allocate a fresh
metadata vector. Evidence tier: value-semantic blocked-QR contract tests plus
empirical component benchmark: `cargo fmt -p hephaestus-wgpu --check`; `cargo
check -p hephaestus-wgpu --bench decomposition_sync`; `cargo nextest run -p
hephaestus-wgpu blocked_qr` (4/4); `cargo bench -p hephaestus-wgpu --bench
decomposition_sync` (QR sync floor 230.962 µs, CPU panel lower bound 28.438 µs,
timestamp total 7.904 µs, median 192 ns). Residual parity gap: no measured QR
speedup from bind-group/metadata hoisting; blocked QR is still dominated by
host/device panel transfer and synchronization.

2026-07-01 (Python/CUDA `spmv_many` surface). Wired the multi-RHS sparse route
through CUDA and `hephaestus-python`: CUDA now exposes `spmv_many`/
`spmv_many_into` as aliases over its sparse-dense kernel, and Python exposes
`hp.spmv_many(...)` across WGPU and CUDA backends. The Rust-side Python sparse
contract now checks `spmv_many` against the existing SpMM output, and the SciPy
parity suite has an external `spmv_many` regression. Evidence tier:
value-semantic in-crate binding test plus static diagnostics: `cargo check -p
hephaestus-cuda`; `cargo check -p hephaestus-python`; `cargo fmt -p
hephaestus-wgpu -p hephaestus-cuda -p hephaestus-python --check`; `cargo
nextest run -p hephaestus-python test_py_sparse_matrix_roundtrip_spmv_spmm`
(1/1). Residual validation gap: the external pytest/CuPy/SciPy suite was
updated but not run in this slice; Python wheel-level parity remains an
environment gate.

2026-07-01 (WGPU `spmv_many` public surface). Added explicit multi-RHS SpMV
APIs (`spmv_many`, `spmv_many_into`, `prepare_spmv_many`) as thin Rust-owned
wrappers over the existing CSR×dense SpMM kernel, making the measured
GPU-preferred route discoverable without duplicating kernels. Contract coverage
checks allocating, caller-owned, and prepared `spmv_many` outputs against the
SpMM reference. Evidence tier: value-semantic WGPU sparse contract plus
empirical API-level benchmark: `cargo fmt -p hephaestus-wgpu --check`; `cargo
check -p hephaestus-wgpu --bench sparse_comparative`; `cargo nextest run -p
hephaestus-wgpu --test contract test_wgpu_sparse_matrix_spmv_spmm` (1/1);
`cargo bench -p hephaestus-wgpu --bench sparse_comparative` (single prepared
SpMV: WGPU 61.146 µs, Leto 1.232 µs; `spmv_many`, 128 RHS vectors: WGPU 62.758
µs, repeated Leto SpMV 150.414 µs; warmed batched prepared SpMM: WGPU 12.258
µs, Leto 35.232 µs). Residual parity gap: single-vector SpMV is still
submit-bound; multi-RHS SpMV now has a named WGPU route above repeated Leto CPU
on the focused benchmark.

2026-07-01 (WGPU batched-SpMV policy evidence). Extended the focused sparse
benchmark with an explicit multi-RHS SpMV regime: Leto CPU runs 128 individual
`spmv` calls per iteration, while WGPU runs the equivalent CSR×dense-RHS
product through prepared SpMM and validates against Leto sparse-dense output.
Evidence tier: differential value check inside the benchmark plus empirical
local timings: `cargo fmt -p hephaestus-wgpu --check`; `cargo check -p
hephaestus-wgpu --bench sparse_comparative`; `cargo bench -p hephaestus-wgpu
--bench sparse_comparative` (single prepared SpMV: WGPU 100.482 µs, Leto 1.232
µs; batched SpMV via SpMM, 128 RHS vectors: WGPU 34.352 µs, repeated Leto SpMV
143.132 µs; warmed batched prepared SpMM: WGPU 10.450 µs, Leto 41.638 µs).
Residual parity gap: single-vector SpMV remains below CPU parity because the
useful work per submit is too small; multi-vector SpMV has a documented GPU
route that exceeds repeated Leto CPU SpMV.

2026-07-01 (WGPU sparse batched-submit amortization). Added
`PreparedSparseDispatch` and `submit_prepared_sparse_batch` so multiple
prepared sparse products can be encoded into one WGPU command buffer and
submitted once. The sparse contract now value-checks batched prepared SpMV and
SpMM outputs against the allocating WGPU/Leto paths. The focused benchmark uses
the fastest measured strategy per operation: prepared one-shot SpMV because the
single-row-work tiny kernel remains submit-bound, and warmed independent-output
batched SpMM because the larger RHS amortizes submission and exceeds Leto CPU
on this local run. Evidence tier: value-semantic WGPU/Leto differential
contract and empirical local benchmark: `cargo fmt -p hephaestus-wgpu
--check`; `cargo check -p hephaestus-wgpu --bench sparse_comparative`; `cargo
nextest run -p hephaestus-wgpu --test contract
test_wgpu_sparse_matrix_spmv_spmm` (1/1); `cargo bench -p hephaestus-wgpu
--bench sparse_comparative` (prepared SpMV: WGPU 65.954 µs, Leto 1.302 µs;
warmed batched prepared SpMM with dense RHS fast path: WGPU 11.940 µs, Leto
38.466 µs). Residual parity gap: SpMV remains below Leto CPU parity at this
small 3-nnz/row shape; SpMM is now faster than Leto in the warmed batched
repeated-dispatch regime.

2026-07-01 (WGPU sparse prepared-dispatch API). Added `prepare_spmv` and
`prepare_spmm` public APIs plus `PreparedSpmv::dispatch` and
`PreparedSpmm::dispatch` so repeated sparse products with fixed buffers reuse
the compiled pipeline, metadata uniform, and bind group instead of rebuilding
them per iteration. Existing `spmv_into`/`spmm_into` remain source-compatible
one-shot paths. The focused sparse comparative benchmark now measures the
prepared path for repeated GPU use. Evidence tier: value-semantic WGPU/Leto
differential contract and empirical local benchmark: `cargo fmt -p
hephaestus-wgpu --check`; `cargo check -p hephaestus-wgpu --bench
sparse_comparative`; `cargo nextest run -p hephaestus-wgpu --test contract
test_wgpu_sparse_matrix_spmv_spmm` (1/1); `cargo bench -p hephaestus-wgpu
--bench sparse_comparative` (prepared SpMV: WGPU 62.636 µs, Leto 1.222 µs;
prepared SpMM with dense RHS fast path: WGPU 48.740 µs, Leto 35.498 µs).
Residual parity gap: SpMM is closer to Leto CPU parity at 0.73x on this local
run; SpMV remains dominated by WGPU submission/poll overhead for this tiny
3-nnz/row workload.

2026-07-01 (WGPU sparse dense-RHS SpMM fast path). Added a C-dense RHS WGPU
SpMM kernel variant that preserves the existing generic strided path for views
while removing signed stride arithmetic from the contiguous-RHS inner loop.
Evidence tier: value-semantic WGPU/Leto differential contract and empirical
local benchmark: `cargo fmt -p hephaestus-wgpu --check`; `cargo check -p
hephaestus-wgpu --bench sparse_comparative`; `cargo nextest run -p
hephaestus-wgpu --test contract test_wgpu_sparse_matrix_spmv_spmm` (1/1);
`cargo bench -p hephaestus-wgpu --bench sparse_comparative` (SpMV reusable
output: WGPU 158.024 µs, Leto 1.484 µs; SpMM reusable output with dense RHS
fast path: WGPU 84.978 µs, Leto 40.752 µs). Residual parity gap: fixed WGPU
submission/synchronization overhead still dominates SpMV and keeps SpMM below
Leto CPU parity at this problem size.

2026-06-30 (WGPU sparse reusable-output parity). Strengthened WGPU CSR sparse
coverage so `spmv_into` and `spmm_into` are value-checked against allocating
`spmv`/`spmm` outputs and prove pre-existing caller-owned output buffers are
overwritten with Leto-matching values. Updated the focused sparse comparative
benchmark timed loops to use reusable caller-owned outputs, so repeated GPU
dispatch is measured without per-iteration output allocation noise. Evidence
tier: value-semantic WGPU/Leto differential contract and static diagnostics:
`cargo nextest run -p hephaestus-wgpu --test contract
test_wgpu_sparse_matrix_spmv_spmm` (1/1); `cargo check -p hephaestus-wgpu
--bench sparse_comparative`; `cargo fmt -p hephaestus-wgpu --check`; `cargo
bench -p hephaestus-wgpu --bench sparse_comparative` (SpMV reusable output:
WGPU 130.940 µs, Leto 1.320 µs; SpMM reusable output: WGPU 88.480 µs, Leto
36.730 µs).

2026-06-30 (CUDA strided-scalar scalar-argument kernel). Replaced
`hephaestus-cuda` strided scalar elementwise lowering through a one-element
device buffer with a dedicated scalar strided CUDA kernel that passes the scalar
as a launch argument. The public scalar API still validates/broadcasts the input
layout through the same strided metadata path and preserves scalar/binary
broadcast semantics. Evidence tier: static diagnostics and value-semantic CUDA
strided contracts on available CUDA runtime: `cargo check -p hephaestus-cuda`;
`cargo fmt -p hephaestus-cuda --check`; `cargo nextest run -p hephaestus-cuda
--test strided` (11/11).

2026-06-30 (Python WGPU/CUDA linalg binding surface). Wired
`hephaestus-python` `Device`, `Array`, and `SparseMatrix` through backend-aware
Rust wrappers so existing Python linalg, sparse, RNG, and elementwise APIs run
on WGPU by default or CUDA via `Device("cuda")` when available. Updated Python
comparative benchmarks to label/select Hephaestus backend explicitly and added
CUDA-backed Python parity coverage plus mixed-backend rejection coverage.
Verified: `cargo check -p hephaestus-python`, `cargo fmt --check`,
`cargo nextest run -p hephaestus-python` (3/3). The bidiagonal factor-contract
blocker is resolved in local Leto by applying reflector-major panels
sequentially when accumulating returned `U`/`V` factors. Verified provider gate:
`cargo nextest run -p leto-ops bidiagonalize_tall` (1/1). Verified Hephaestus
gate: `cargo nextest run -p hephaestus-wgpu --test contract` (94/94).

2026-06-26 (CUDA dynamic-rank strided delegation). Added
`hephaestus-cuda` dynamic-rank strided elementwise entry points over borrowed
shape/stride slices so runtime-shaped consumers can delegate strided CUDA
binary/unary kernels without materializing fixed-rank Leto layouts or carrying
duplicate local PTX generators. Static-rank APIs now share the same private
binary/unary launch helpers. Coeus routes rank <= 4 supported strided primitive
ops through this provider surface and retains local kernels only for unsupported
activations, write-aliasing, or wider-rank layouts. Verified:
`cargo check -p hephaestus-cuda`, `cargo fmt -p hephaestus-cuda --check`,
`cargo clippy -p hephaestus-cuda --all-targets -- -D warnings`, `cargo doc -p
hephaestus-cuda --no-deps`, `cargo nextest run -p hephaestus-cuda --test
strided` (11/11), and downstream `cargo nextest run -p coeus-cuda --features
cuda` (69/69).

2026-06-23 (strided-scalar pooled-uniform kernel). Paranoid re-audit of the hot
per-op kernels (reduction, scan, strided, sparse) via two skeptical agents.
Verified clean: the multi-pass reduction correctly encodes all passes in one
submit (the `temp_buffers` retention is *required*, not waste); scan/strided/
sparse meta uniforms are pooled; no `poll(Wait)` over-sync; Pod casts are
size-correct; SpMV/SpMM column-index bounds are guaranteed by Leto's CsrMatrix
construction. Found and fixed one real inconsistency: the strided scalar path
allocated a per-call device storage buffer for the broadcast scalar (the
contiguous path already uses a pooled uniform). Added a dedicated
`StridedScalarKernel` reading the scalar from a pooled uniform — no per-call
storage operand, value-identical (verified). Recorded the storage-pool
examination (deferred: within-call buffers are submit-pinned, cross-call needs
fences). The CUDA strided-scalar follow-on is closed by the 2026-06-30
scalar-argument kernel entry above.
Verified: `cargo fmt`, `cargo clippy -p hephaestus-wgpu --all-targets -- -D
warnings`, strided/scalar contract tests, full workspace nextest, doctests.

2026-06-23 (blocked-decomposition host-allocation reuse). Paranoid memory/safety
re-audit of the WGPU decomposition modules. Verified the Pod meta structs
(`SyrkMeta`/`GemmMeta`/`HhMeta`/`HhReflectorMeta`/`RegionCopyMeta`) are all
`#[repr(C)]`, padding-free, with true SAFETY comments, and the transfer/pool core
(`stage_and_read`, sub-buffer paths) is bounds/overflow-checked and RAII-pooled —
clean. Found and fixed real memory churn: the blocked Cholesky/LU/QR panel loops
allocated a fresh host `Vec` per panel for each region download plus per-panel
scratch. Added the region-download SSOT `download_matrix_region_compact_into`
(reuses host capacity), removed the dead returning-`Vec` wrapper, and hoisted each
decomposition's per-panel host buffers above the loop. Verified: `cargo fmt`,
`cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`, blocked LU/QR/
Cholesky cross-boundary contract tests, full workspace nextest, doctests.

2026-06-23 (residual-gap review). Reviewed the residual register and resolved the
two genuinely-actionable gaps: `matrix_rank`/`det` ill-conditioned divergence was
previously documented-but-untested at the threshold boundary. Documented both
contracts on the public APIs (`matrix_rank` relative-threshold pivot criterion;
`det` no-determinant-tolerance pivot product) and added analytically-derived
contract tests (`matrix_rank_relative_tolerance_is_the_discriminator`,
`det_of_near_singular_triangular_is_exact_pivot_product`). Restructured
`gap_audit.md` into an honest SSOT (Resolved / Accepted design / Open future work
/ Environment) so the remaining residuals — host-delegated GPU-kernel/performance
parity ([major], tracked) and CUDA hardware/toolchain blockers — are no longer
conflated with open defects. Verified: `cargo fmt`, `cargo clippy
-p hephaestus-wgpu -p hephaestus-core --all-targets -- -D warnings`, focused
rank/det tests, doctests, full workspace nextest.

2026-06-23 (WGPU staging-registry contention). Audited the staging-allocator
integration in `infrastructure/{device,buffer,pool}.rs`. Found the global
`WGPU_MAPPED_BUFFERS` registry resolved a sub-allocated staging pointer to its
mapped block with an `O(n)` `.iter().find()` scan held inside the global lock at
both HostPinned alloc/upload sites. Replaced the `HashMap` with a base-address
`BTreeMap` and a single `resolve_mapped_buffer` helper doing an `O(log n)`
`range(..=ptr).next_back()` containment query (DRY/SSOT). Tightened the registry
+ descriptor to `pub(crate)` (no external consumers; makes the type change
non-breaking) and removed the dead `usage` field. Verified: `cargo fmt`, `cargo
clippy -p hephaestus-wgpu -p hephaestus-core --all-targets -- -D warnings`,
`cargo nextest run --workspace` (228 passed incl. `test_placement_aware_allocation`,
upload/download round-trip), doctests. Remaining bounded characteristic recorded
in gap_audit: the registry still uses one global `Mutex` (now `O(log n)`); a
sharded registry is deferred until a workload measures the lock as hot.

Target version: 0.10.0 (bumped; CHANGELOG synced). Sprint phase: Execution.
Phase 1 COMPLETE. Phase 2 current increment: `hephaestus-wgpu` Leto parity
linalg and comparative benchmarks. Next concrete increment: complete WGPU/Leto
parity audit for remaining operator families and shared Atlas seam usage
(`mnemosyne`, `moirai`, `themis`, `hermes`).

Latest verification: `cargo fmt --check`,
`cargo clippy --workspace --all-targets --locked -- -D warnings`,
`cargo nextest run --workspace --locked` (229 passed),
`cargo test --doc --workspace --locked`,
`cargo doc --workspace --no-deps --locked`,
`cargo metadata --no-deps --locked --format-version 1`, `git diff --check`,
and `cargo bench -p hephaestus-wgpu --bench sparse_comparative --locked`.
`cargo semver-checks --workspace --all-features` is blocked because
`hephaestus-core` has no crates.io baseline.

## Unreleased WGPU Leto parity linalg [minor]
- [x] Added GPU-resident allocating `matmul`/`batched_matmul` and caller-owned
  `matmul_into`/`batched_matmul_into`, plus `dot`, `trace`, `norm_l1`,
  `norm_l2`, and `norm_max` over strided operands.
- [x] Added GPU-resident allocating `kron` and caller-owned `kron_into` over
  strided matrix operands, with Leto differential contract coverage and
  comparative benchmark coverage against Leto, `ndarray`, and a
  nalgebra-backed reference implementation.
- [x] Added GPU-resident `matpow` over strided square matrix operands, using
  exponentiation by squaring over WGPU `matmul_into` dispatches. Differential
  tests cover Leto parity for floating-point shear powers, integer `A^0`, and
  non-square rejection; comparative benchmarks cover Leto, an `ndarray`
  repeated-squaring reference, and `nalgebra`.
- [x] Added GPU-resident finite-`f32` `matrix_rank` and
  `matrix_rank_with_tolerance` over strided rank-2 operands. Differential tests
  cover exact finite full-rank, rank-deficient, and zero matrices against Leto,
  plus empty-matrix rejection; comparative benchmarks cover WGPU, Leto,
  `ndarray`-backed, and `nalgebra`-backed references. Residual distinction:
  WGPU uses row-reduction pivots while Leto's rank uses the SVD spectrum.
- [x] Added GPU-resident finite-`f32` `det` over strided square rank-2
  operands using the shared WGPU matrix-property row-reduction dispatch.
  Differential tests cover exact finite nonsingular and singular matrices
  against Leto plus rectangular rejection; comparative benchmarks cover WGPU,
  Leto, `ndarray`, and `nalgebra` references. Residual distinction: WGPU uses
  exact row-reduction pivots with no tolerance for determinant while Leto uses
  its CPU determinant algorithm.
- [x] Added WGPU device-resident Cholesky, LU, and QR decomposition surfaces
  mirroring Leto's decomposition, solve, determinant, and inverse APIs where
  each factorization supports them. Differential tests compare factors and
  solve/inverse outputs against Leto; comparative benchmarks cover WGPU, Leto,
  and `nalgebra`. Residual distinction: factorization currently delegates to
  Leto on the host and uploads factors to the device, so this is API parity and
  measured transfer/host-factorization overhead, not GPU-kernel parity.
- [x] Added WGPU device-resident SVD parity coverage. Contract tests cover
  closed-form singular values, thin-SVD reconstruction against Leto, and
  rank-revealing behavior on a rank-deficient matrix; comparative benchmarks
  cover WGPU, Leto, and `nalgebra` SVD on a 32x16 full-column-rank matrix.
- [x] Added WGPU device-resident bidiagonalization parity coverage. Contract
  tests cover orthogonal `U`/`V`, upper-bidiagonal structure, `U B V^T`
  reconstruction, singular-value preservation, and wide-matrix rejection;
  comparative benchmarks cover WGPU, Leto, and `nalgebra` SVD on a 32x16
  matrix.
- [x] Added WGPU device-resident Schur parity coverage. Contract tests cover
  orthogonal `Q`, quasi-upper-triangular `T`, `Q T Q^T` reconstruction,
  spectrum preservation, and rectangular rejection; comparative benchmarks
  cover WGPU, Leto, and `nalgebra` complex eigenvalues on a 32x32 block
  matrix.
- [x] Added WGPU device-resident Hessenberg parity coverage. Contract tests
  cover orthogonal `Q`, upper-Hessenberg `H`, `Q H Q^T` reconstruction,
  trace/norm similarity invariants, and rectangular rejection; comparative
  benchmarks cover WGPU, Leto, and `nalgebra` Hessenberg reduction on a 32x32
  matrix.
- [x] Added WGPU device-resident full-pivot LU parity coverage. Contract tests
  cover packed `L/U` reconstruction of `P A Q`, rank reporting, determinant,
  solve, inverse, rank-deficient inverse rejection, and rectangular rejection;
  comparative benchmarks cover WGPU, Leto, and `nalgebra` full-pivot LU on a
  32x32 matrix.
- [x] Added WGPU device-resident Bunch-Kaufman parity coverage. Contract tests
  cover downloaded `L`/`D` factors, permutation agreement with Leto,
  `L D L^T = P A P^T` reconstruction, rectangular rejection, and
  nonsymmetric rejection; comparative benchmarks cover WGPU and Leto
  Bunch-Kaufman with `nalgebra` determinant as the external CPU comparator.
- [x] Added WGPU device-resident UDU parity coverage. Contract tests cover
  downloaded `U`/diagonal `D` factors, determinant, solve, inverse,
  `U D U^T` reconstruction of a symmetric indefinite matrix, rectangular
  rejection, nonsymmetric rejection, and zero-pivot rejection; comparative
  benchmarks cover WGPU and Leto UDU with `nalgebra` determinant as the
  external CPU comparator.
- [x] Added WGPU device-resident column-pivoted QR, pseudoinverse, and matrix
  exponential baseline parity coverage. Contract tests cover column-pivoted QR
  factor agreement against Leto plus closed-form diagonal pseudoinverse and
  matrix exponential cases; comparative benchmarks cover WGPU and Leto for all
  three, with `nalgebra` comparators for column-pivoted QR and pseudoinverse.
- [x] Strengthened WGPU pseudoinverse and matrix exponential parity coverage.
  Contract tests now cover rank-deficient Moore-Penrose identities,
  rectangular full-rank pseudoinverse, non-finite pseudoinverse rejection,
  nilpotent and skew-symmetric matrix-exponential closed forms, a general
  `nalgebra` matrix-exponential oracle, and rectangular/non-finite matrix
  exponential rejection.
- [x] Added WGPU device-resident symmetric Jacobi eigen decomposition and
  eigenvalues-only surfaces mirroring Leto. Differential tests compare
  eigenvalues and eigenvectors against Leto and reject non-symmetric inputs;
  comparative benchmarks cover WGPU, Leto, and `nalgebra` eigenvalues with
  order-insensitive cross-library eigenvalue comparison. Residual distinction:
  eigensolve currently delegates to Leto on the host and uploads results to the
  device, so this is API parity and measured transfer/host-eigensolve overhead,
  not GPU-kernel eigensolver parity.
- [x] Added WGPU device-resident general eigenvalues over complex output
  buffers for square `f32` matrices. Contract coverage includes a diagonal
  matrix with closed-form real eigenvalues and a nonsymmetric Leto
  differential case; comparative benchmark coverage now measures a 32x32
  block-rotation matrix against Leto and `nalgebra` complex eigenvalues.
- [x] Strengthened WGPU general-eigenvalue contract coverage. Tests now cover
  exact 2x2 and 3x3 complex-pair blocks, triangular spectra, structured real
  nonsymmetric spectra, dense 5x5 `nalgebra` oracle comparison,
  symmetric-input all-real spectra, unordered spectrum matching, and
  rectangular rejection.
- [x] Added blocked WGPU Cholesky entry point with CPU panel factorization and
  triangular solve plus GPU SYRK trailing update. Differential coverage now
  includes a 66x66 SPD matrix crossing the 64-wide block boundary; comparative
  benchmarks now measure 128x128 blocked Cholesky against Leto and `nalgebra`.
- [x] Added comparative benchmark rows for blocked WGPU LU and QR paths,
  validating blocked LU solve parity and blocked QR factor parity before
  timing them against Leto and `nalgebra`.
- [x] Routed WGPU launch planning through Mnemosyne `KernelResourceBudget` and
  Moirai GPU `plan_launch` while preserving Hephaestus checked overflow
  semantics from `BlockWidth::checked_covering_blocks`.
- [x] Documented Atlas compute-boundary integration for Mnemosyne, Moirai,
  Themis, and Hermes in ADR 0002 and README. Hermes is integrated at the
  host-delegated Leto tier via `leto-ops`' `simd` feature; direct WGPU/CUDA
  kernel calls into Hermes are rejected as a boundary violation because Hermes
  owns CPU SIMD over host slices, not GPU shader/PTX execution.
- [x] Switched Hephaestus' `moirai-gpu` dependency to Moirai's planner-only
  feature set, closing the duplicate WGPU runtime dependency while preserving
  Mnemosyne `KernelResourceBudget` + Moirai `plan_launch` dispatch sizing.
- [x] Added GPU-resident rank-2 `reduce_axis`, `sum_axis`, `min_axis`,
  `max_axis`, `mean_axis`, and caller-owned `*_axis_into` forms, preserving
  Leto's rank-preserving axis-reduction contract (`[rows, cols] -> [1, cols]`
  or `[rows, 1]`). Differential tests cover caller-owned and allocating sum,
  min, max, and mean against Leto; comparative benchmarks cover axis 0.
- [x] Added GPU-resident rank-2 `scan_axis_into`, `scan_axis`,
  `cumsum_into`, and `cumsum`, with forward/reverse scan direction and
  cumulative sum/product markers. Differential tests cover caller-owned and
  allocating Cumsum plus reverse cumulative product against Leto; comparative
  benchmarks cover Cumsum over axis 1 against Leto, an `ndarray` reference,
  and a nalgebra-backed reference.
- [x] Added allocating strided elementwise wrappers
  `binary_elementwise_strided`, `unary_elementwise_strided`, and
  `scalar_elementwise_strided`, returning C-contiguous GPU buffers while
  delegating to the existing caller-owned strided kernels. Differential tests
  cover allocated binary, unary, and scalar outputs against the same CPU
  references used for caller-owned dispatch.
- [x] Corrected `norm_l2` to return `sqrt(sum(x*x))`, matching Leto's CPU
  contract rather than exposing the squared-sum intermediate.
- [x] Extended `comparative` benchmark coverage to WGPU vs Leto, `ndarray`,
  and `nalgebra`; refreshed `benchmark_results.md` from a real local WGPU run
  including blocked 128x128 Cholesky.
- [x] Added fused WGPU map-reduction dispatch for trace and L1 norm. Dot
  product, L2 norm, and max norm retain the measured faster staged paths after
  the fused variant regressed in the local comparative run.
- [x] Replaced WGPU CSR SpMV and SpMM host-delegated products with real WGSL
  kernels over device-resident CSR values, packed CSR index buffers, and
  device-resident dense operands/results. The `sparse` feature owns the Leto
  CSR upload/download boundary, while dispatch sizing flows through the shared
  Mnemosyne `KernelResourceBudget` and Moirai `plan_launch` helper.
- Evidence: `cargo fmt -p hephaestus-wgpu -p hephaestus-cuda --check`;
  `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`; `cargo
  nextest run -p hephaestus-wgpu -j 1` (62 passed); `cargo nextest run -p
  hephaestus-cuda -j 1` (51 passed); `cargo test --doc -p hephaestus-wgpu` (0
  doctests); `cargo test --doc -p hephaestus-cuda` (0 doctests); `cargo doc -p
  hephaestus-wgpu --no-deps`; `cargo doc -p hephaestus-cuda --no-deps`; `cargo
  bench -p hephaestus-wgpu --bench comparative` (refreshed
  `benchmark_results.md`, including blocked 128x128 Cholesky, matrix rank,
  determinant, LU, full-pivot LU, QR, SVD, bidiagonalization, Schur,
  Hessenberg, Bunch-Kaufman, UDU, column-pivoted QR, pseudoinverse, matrix
  exponential, symmetric eigen, and general eigenvalues; CUDA rows skipped because the WGPU bench depends on
  `hephaestus-cuda` without its
  `cuda` feature in this environment); `git diff --check`. Full workspace
  all-features clippy
  attempted earlier and blocked before this slice by `cuda-bindings` requiring
  `CUDA_TOOLKIT_PATH`. Evidence tier: value-semantic differential tests,
  static diagnostics, and empirical benchmarks.
- Additional matrix-function evidence: `cargo nextest run -p hephaestus-wgpu
  linalg_pinv linalg_matexp -j 1 --no-fail-fast --no-capture` (8 passed);
  `cargo clippy -p hephaestus-wgpu --test contract -- -D warnings`;
  `rustfmt --edition 2021 --check crates/hephaestus-wgpu/tests/contract.rs`;
  `git diff --check`. Evidence tier: value-semantic closed-form,
  Moore-Penrose algebraic, differential, and invalid-input tests.
- Additional general-eigenvalue evidence: `cargo nextest run -p
  hephaestus-wgpu eigenvalues -j 1 --no-fail-fast --no-capture` (6 passed);
  `cargo clippy -p hephaestus-wgpu --test contract -- -D warnings`;
  `rustfmt --edition 2021 --check crates/hephaestus-wgpu/tests/contract.rs`;
  `git diff --check`. Evidence tier: closed-form, differential, and
  invalid-input value-semantic tests.
- Atlas compute-boundary evidence: `docs/adr/0002-atlas-compute-boundaries.md`,
  README layer-boundary update, `leto-ops` dependency audit confirming the
  `simd` feature and Hermes dispatch calls, and Hermes README/backlog boundary
  audit. Evidence tier: implementation and documentation audit.
- Additional blocked decomposition benchmark evidence: `cargo nextest run -p
  hephaestus-wgpu blocked_lu blocked_qr -j 1 --no-fail-fast` (8 passed);
  `cargo check -p hephaestus-wgpu --bench comparative`; `cargo bench -p
  hephaestus-wgpu --bench comparative` (added blocked LU/QR timing rows).
- Additional blocked decomposition synchronization evidence: `cargo check -p
  hephaestus-wgpu --bench decomposition_sync`; `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (LU 66x66 sync floor 308.2 µs,
  QR 70x35 sync floor 227.8 µs).
- Additional blocked LU transfer evidence: narrowed blocked LU host/device
  transfers to the active diagonal-panel and trailing-submatrix regions.
  `cargo nextest run -p hephaestus-wgpu blocked_lu -j 1 --no-fail-fast` (4
  passed); `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`;
  `cargo bench -p hephaestus-wgpu --bench comparative` (blocked LU 66x66
  239.8 µs, Leto 62.1 µs, nalgebra 7.0 µs; blocked QR 70x35 1.13 ms, Leto
  11.4 µs, nalgebra 6.1 µs); `cargo bench -p hephaestus-wgpu --bench
  decomposition_sync` (LU 66x66 sync floor 405.2 µs, QR 70x35 sync floor
  240.7 µs). Evidence tier: value-semantic blocked LU differential tests,
  static diagnostics, and empirical local benchmarks.
- Additional blocked QR transfer evidence: factored LU region transfers into
  a decomposition-local helper and changed blocked QR to upload/download a
  compact trailing-column tile per panel before applying Householder
  reflectors. `cargo fmt -p hephaestus-wgpu --check`; `cargo clippy -p
  hephaestus-wgpu --all-targets -- -D warnings`; `cargo nextest run -p
  hephaestus-wgpu blocked_lu blocked_qr -j 1 --no-fail-fast` (8 passed);
  `cargo bench -p hephaestus-wgpu --bench decomposition_sync` (LU 66x66 sync
  floor 220.7 µs, QR 70x35 sync floor 205.6 µs); `cargo bench -p
  hephaestus-wgpu --bench comparative` (blocked LU 66x66 260.5 µs, Leto 63.5
  µs, nalgebra 7.1 µs; blocked QR 70x35 1.04 ms, Leto 10.4 µs, nalgebra 6.0
  µs). Evidence tier: value-semantic blocked LU/QR differential tests, static
  diagnostics, and empirical local benchmarks.
- Additional blocked QR reflector-upload evidence: packed all panel
  Householder vectors into one device buffer and extended the reflector kernel
  metadata with a vector offset, removing per-reflector vector-buffer creation
  while preserving one GPU launch per reflector. `cargo fmt -p hephaestus-wgpu
  --check`; `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`;
  `cargo nextest run -p hephaestus-wgpu blocked_lu blocked_qr -j 1
  --no-fail-fast` (8 passed); `cargo bench -p hephaestus-wgpu --bench
  decomposition_sync` (LU 66x66 sync floor 219.9 µs, QR 70x35 sync floor
  138.5 µs); `cargo bench -p hephaestus-wgpu --bench comparative` (blocked LU
  66x66 273.7 µs, Leto 66.2 µs, nalgebra 7.1 µs; blocked QR 70x35 1.05 ms,
  Leto 10.5 µs, nalgebra 6.1 µs). Evidence tier: value-semantic blocked
  LU/QR differential tests, static diagnostics, and empirical local
  benchmarks.
- Additional blocked QR timestamp evidence: extended `decomposition_sync` with
  a `wgpu::TIMESTAMP_QUERY` path that creates a timestamp-capable device when
  available, wraps 32 minimal compute passes with begin/end timestamps, and
  resolves them to host memory. `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/benches/decomposition_sync.rs`; `cargo clippy -p
  hephaestus-wgpu --all-targets -- -D warnings`; `cargo nextest run -p
  hephaestus-wgpu blocked_lu blocked_qr -j 1 --no-fail-fast` (8 passed);
  `cargo bench -p hephaestus-wgpu --bench decomposition_sync` (LU 66x66 sync
  floor 338.4 µs, QR 70x35 sync floor 207.6 µs, QR 32-reflector timestamp
  launch total 155.2 µs, median 3.4 µs). Evidence tier: value-semantic blocked
  LU/QR tests, static diagnostics, synthetic transfer benchmark, and
  GPU-timeline timestamp-query measurement.
- Additional blocked QR reflector-batching evidence: changed the WGPU
  Householder panel kernel so one compute pass owns a trailing column and
  applies all panel reflectors sequentially inside that workgroup, eliminating
  per-reflector compute-pass launches while preserving reflector order.
  `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/src/application/decomposition/qr.rs`; `cargo clippy
  -p hephaestus-wgpu --all-targets -- -D warnings`; `cargo nextest run -p
  hephaestus-wgpu blocked_qr -j 1 --no-fail-fast` (4 passed); `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (QR timestamp total 8.4 µs,
  median 160 ns); `cargo bench -p hephaestus-wgpu --bench comparative`
  (blocked QR 70x35 420.8 µs, Leto 10.7 µs, nalgebra 6.1 µs). Evidence tier:
  value-semantic blocked QR tests, static diagnostics, empirical benchmark,
  and GPU-timeline timestamp-query measurement.
- Additional blocked QR component-profile evidence: extended
  `decomposition_sync` to measure the 70x35 CPU panel-factorization lower
  bound and removed the obsolete final-Leto-recompute profile row. The
  production path builds the host-side `QrDecomposition` from computed blocked
  factors with `from_raw_parts`.
  `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/benches/decomposition_sync.rs`; `cargo check -p
  hephaestus-wgpu --bench decomposition_sync`; `cargo clippy -p
  hephaestus-wgpu --bench decomposition_sync -- -D warnings`; `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (LU sync floor 359.6 µs, QR sync
  floor 222.6 µs, QR CPU panel lower bound 26.3 µs, QR timestamp total 8.2 µs,
  median 160 ns). Evidence tier: static
  diagnostics, empirical component benchmark, and GPU-timeline timestamp-query
  measurement.
- Additional blocked QR metadata-transfer evidence: packed the panel
  Householder vector offsets and beta coefficients into one
  `HhReflectorMeta` storage buffer, replacing two per-panel metadata uploads
  and bindings with one while preserving reflector order and arithmetic.
  `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/src/application/decomposition/qr.rs`; `cargo check
  -p hephaestus-wgpu`; `cargo clippy -p hephaestus-wgpu --all-targets --
  -D warnings`; `cargo nextest run -p hephaestus-wgpu blocked_qr -j 1
  --no-fail-fast` (4 passed); `cargo bench -p hephaestus-wgpu --bench
  decomposition_sync` (QR sync floor 219.9 µs, CPU panel lower bound 25.3 µs,
  timestamp total 8.3 µs, median 192 ns);
  `cargo bench -p hephaestus-wgpu --bench comparative` (blocked QR 70x35
  480.8 µs, Leto 14.9 µs, nalgebra 10.0 µs). Evidence tier:
  value-semantic blocked QR tests, static diagnostics, empirical benchmark,
  and GPU-timeline timestamp-query measurement.
- Additional blocked QR WGPU-resource reuse evidence: hoisted the Householder
  metadata uniform and bind group out of the panel loop and reused host
  reflector metadata scratch. `cargo fmt -p hephaestus-wgpu --check`; `cargo
  check -p hephaestus-wgpu --bench decomposition_sync`; `cargo nextest run -p
  hephaestus-wgpu blocked_qr` (4 passed); `cargo bench -p hephaestus-wgpu
  --bench decomposition_sync` (QR sync floor 230.962 µs, CPU panel lower bound
  28.438 µs, timestamp total 7.904 µs, median 192 ns). Evidence tier:
  value-semantic blocked QR tests, static diagnostics, empirical component
  benchmark, and GPU-timeline timestamp-query measurement.
- Additional blocked QR delayed-copy evidence: first panel downloads from the
  original input buffer before the full copy to `work_buf` is queued, allowing
  the copy to overlap first-panel CPU factorization while queue ordering
  preserves later `work_buf` writes. `cargo fmt -p hephaestus-wgpu --check`;
  `cargo check -p hephaestus-wgpu --bench decomposition_sync`; `cargo nextest
  run -p hephaestus-wgpu blocked_qr` (4 passed); `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (QR sync floor 213.209 µs, CPU
  panel lower bound 26.369 µs, timestamp total 7.776 µs, median 192 ns).
  Evidence tier: value-semantic blocked QR tests, static diagnostics, empirical
  component benchmark, and GPU-timeline timestamp-query measurement.

## Unreleased CUDA Leto parity application surface [minor]
- [x] CUDA exports mirror the current WGPU/Leto core operation and decomposition slice:
  elementwise, strided elementwise, reductions, rank-2 axis reductions,
  rank-2 scans, `cumsum_into`/`cumsum`, matrix multiplication, Kronecker
  product, matrix power, finite-`f32` matrix rank, dot, trace, norms, and
  Cholesky/LU/QR decompositions.
- [x] Renamed CUDA forward cumulative-sum caller-owned API to `cumsum_into`,
  matching Leto and WGPU with no compatibility alias.
- [x] Stub-mode CUDA build validates the operation surface without fabricating
  hardware: unavailable-device tests skip by construction, while contract tests
  still exercise host-visible error paths and CPU-backed semantics available in
  the stub.
- [x] Removed stale default-build CUDA blocked-Cholesky export/test references
  because the CUDA blocked SYRK path is CUDA-feature gated and not verified in
  the default stub build.
- Additional WGPU sparse evidence: `cargo check -p hephaestus-wgpu`; `cargo
  clippy -p hephaestus-wgpu --lib -- -D warnings`; `rustfmt --edition 2021
  --check` over the WGPU sparse/module/export files; `cargo nextest run -p
  hephaestus-wgpu sparse -j 1 --no-fail-fast` (1 passed);
  `cargo check -p hephaestus-wgpu --bench sparse_comparative`;
  `cargo bench -p hephaestus-wgpu --bench sparse_comparative` (latest prepared
  SpMV 1000x1000 CSR reusable output: WGPU 61.146 µs, Leto 1.232 µs; latest
  `spmv_many` with 128 RHS vectors: WGPU 62.758 µs, repeated Leto SpMV
  150.414 µs; latest warmed batched prepared SpMM 1000x1000x128 reusable output
  with dense RHS fast path: WGPU 12.258 µs, Leto 35.232 µs). Evidence tier: static diagnostics,
  value-semantic GPU sparse contract test, value-checked benchmark outputs,
  and empirical local benchmark.
- Evidence: `cargo fmt -p hephaestus-wgpu -p hephaestus-cuda --check`; `cargo
  clippy -p hephaestus-wgpu --all-targets -- -D warnings` (compiles
  `hephaestus-cuda` as the WGPU dev dependency); `cargo nextest run -p
  hephaestus-cuda -j 1` (51 passed); `cargo test --doc -p hephaestus-cuda` (0
  doctests); `cargo doc -p hephaestus-cuda --no-deps`. Evidence tier: static
  diagnostics and value-semantic contract tests in the currently available
  stub mode.

## 0.10.0 checked launch grid arithmetic [minor]
- [x] Added `BlockWidth::checked_covering_blocks` as the non-saturating launch
  grid arithmetic API for backends that need typed dispatch errors.
- [x] Routed WGPU `workgroups` through the checked core API, leaving WGPU only
  responsible for converting `None` into `HephaestusError::DispatchFailed`.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-core
  domain::launch::tests --offline` (2 passed); `cargo test -p
  hephaestus-wgpu application::pipeline::tests --offline` (2 passed);
  `cargo check --workspace --offline`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (35 passed); `cargo test --doc
  --workspace --locked`; `cargo metadata --no-deps --locked --format-version
  1`; `cargo doc --workspace --no-deps --locked`; `cargo bench --bench
  elementwise_into --locked` on real adapter (allocating 332,480 ns/iter;
  caller-owned 102,150 ns/iter for 1,048,576 elements, 20 iterations);
  `cargo bench --bench reduction_width --locked` on real adapter (default
  42,960 ns/iter; width-128 91,620 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, dispatch contract tests,
  static diagnostics, and empirical benchmarks.

## 0.9.4 WGPU byte-size SSOT [patch]
- [x] Made the checked byte-size helper available to WGPU application modules.
- [x] Replaced the remaining local `size_of::<...>() as u64` buffer-size
  calculations in scalar uniform acquisition, strided metadata uniform
  acquisition, and singleton reduction copy encoding.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (3 passed); `cargo test -p
  hephaestus-wgpu application::reduction::tests --offline` (1 passed);
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (35
  passed); `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  206,970 ns/iter; caller-owned 60,640 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 47,335 ns/iter; width-128 55,895 ns/iter for 65,536 elements, 20
  iterations); `rg` confirmed no remaining local `size_of::<...>() as u64`
  buffer-size casts in `crates/hephaestus-wgpu/src`. Deeper gate attempted:
  `cargo semver-checks --workspace --all-features` blocked because
  `hephaestus-core` is not published in the registry. Evidence tier:
  value-semantic unit tests, dispatch contract tests, static diagnostics, and
  empirical benchmarks.

## 0.9.3 upload byte-size precheck [patch]
- [x] Routed `WgpuDevice::upload` through the shared checked byte-size helper
  before `create_buffer_init`, so upload, allocation, and download paths use
  the same allocation-overflow boundary.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (3 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (35 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  244,285 ns/iter; caller-owned 59,270 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 41,100 ns/iter; width-128 58,235 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, dispatch contract tests,
  static diagnostics, and empirical benchmarks.

## 0.9.2 dispatch precheck completion [patch]
- [x] Hoisted binary and unary dispatch workgroup-range validation before
  pipeline cache lookup, bind-group creation, and command encoding.
- [x] Hoisted reduction workgroup-range validation before intermediate output
  buffer allocation in each reduction pass.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  application::pipeline::tests --offline` (2 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (35 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  239,850 ns/iter; caller-owned 152,440 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (rerun: default 44,580 ns/iter; width-128 155,995 ns/iter for 65,536
  elements, 20 iterations). Deeper gate attempted: `cargo semver-checks
  --workspace --all-features` blocked because `hephaestus-core` is not
  published in the registry. Evidence tier: value-semantic unit tests,
  dispatch contract tests, static diagnostics, and empirical benchmarks.

## 0.9.1 dispatch range precheck [patch]
- [x] Hoisted scalar dispatch workgroup-range validation before transient
  uniform-buffer acquisition.
- [x] Hoisted strided dispatch workgroup-range validation before transient
  metadata uniform-buffer acquisition.
- [x] Added shared `workgroups` boundary coverage for the exact `u32::MAX`
  workgroup limit and one element beyond it.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  application::pipeline::tests --offline` (2 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (35 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  322,175 ns/iter; caller-owned 89,460 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 38,700 ns/iter; width-128 41,930 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, dispatch contract tests,
  static diagnostics, and empirical benchmarks.

## 0.9.0 transient buffer alignment [minor]
- [x] Added a shared checked `aligned_size` helper for WGPU byte alignment.
- [x] Made `get_staging_buffer` and `get_uniform_buffer` return
  `Result<wgpu::Buffer>` and reject alignment overflow with
  `AllocationFailed`.
- [x] Updated scalar, strided, and download call sites to propagate allocation
  failures.
- [x] Added value-semantic unit coverage for alignment overflow.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (3 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (33 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (serial
  rerun: allocating 263,335 ns/iter; caller-owned 70,295 ns/iter for
  1,048,576 elements, 20 iterations); `cargo bench --bench reduction_width
  --locked` on real adapter (serial rerun: default 47,740 ns/iter; width-128
  107,070 ns/iter for 65,536 elements, 20 iterations). Deeper gate attempted:
  `cargo semver-checks --workspace --all-features` blocked because
  `hephaestus-core` is not published in the registry. Evidence tier:
  value-semantic unit tests, dispatch contract tests, static diagnostics, and
  empirical benchmarks.

## 0.8.1 pipeline cache critical section [patch]
- [x] Split `cached_pipeline` into a locked cache-hit check, unlocked WGPU
  pipeline compilation, and locked recheck/insert.
- [x] Preserved cache correctness under races by rechecking the key before
  insertion.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; `cargo nextest run --workspace
  --locked` (32 passed); `cargo test --doc --workspace --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo doc
  --workspace --no-deps --locked`; `cargo bench --bench elementwise_into
  --locked` on real adapter (allocating 383,335 ns/iter; caller-owned 74,275
  ns/iter for 1,048,576 elements, 20 iterations); `cargo bench --bench
  reduction_width --locked` on real adapter (default 56,090 ns/iter;
  width-128 59,900 ns/iter for 65,536 elements, 20 iterations). Evidence tier:
  value-semantic dispatch contract tests, static diagnostics, and empirical
  benchmark.

## 0.8.0 checked allocation sizing [minor]
- [x] Added `HephaestusError::AllocationFailed` as the typed boundary for
  allocation requests rejected before buffer creation.
- [x] Replaced unchecked WGPU byte-size multiplication with checked exact and
  padded size helpers shared by allocation and download sizing.
- [x] Added unit coverage for copy-alignment padding and overflow rejection
  without allocating memory.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (2 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (32 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench reduction_width --locked` on real adapter (default
  33,110 ns/iter; width-128 49,195 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, contract tests, static
  diagnostics, and empirical benchmark.

## 0.7.3 reduction pass storage [patch]
- [x] Added a single `reduction_pass_count` helper for the multi-pass tree
  depth calculation.
- [x] Preallocated the intermediate `WgpuBuffer` handle vector with that pass
  count before command encoding.
- [x] Added value-semantic unit coverage for empty, singleton, exact-width,
  trailing-width, and multi-pass depths.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  application::reduction::tests::pass_count_matches_tree_depth --offline`
  (1 passed); `cargo check --workspace --locked`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; `cargo nextest run --workspace
  --locked` (30 passed); `cargo test --doc --workspace --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo doc
  --workspace --no-deps --locked`; `cargo bench --bench reduction_width
  --locked` on real adapter (rerun: default 50,330 ns/iter; width-128
  97,035 ns/iter for 65,536 elements, 20 iterations). Evidence tier:
  value-semantic unit tests, contract tests, static diagnostics, and empirical
  benchmark.

## 0.7.2 reduction width validation [patch]
- [x] Moved `reduction_with_width` power-of-two validation before empty and
  singleton fast paths.
- [x] Added boundary contract coverage proving invalid widths are rejected for
  empty, singleton, and multi-element inputs.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo nextest run -p hephaestus-wgpu
  reduction_width_is_part_of_dispatch_contract --locked` (1 passed);
  `cargo check --workspace --locked`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; `cargo nextest run --workspace
  --locked` (29 passed); `cargo test --doc --workspace --locked`;
  `cargo doc --workspace --no-deps --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo bench --bench reduction_width
  --locked` on real adapter (default 49,945 ns/iter; width-128 55,945
  ns/iter for 65,536 elements, 20 iterations); `git diff --check`. Evidence
  tier: value-semantic contract tests, static diagnostics, and empirical
  benchmark.

## 0.7.1 reduction-width benchmark [patch]
- [x] Added `reduction_width` benchmark target for default vs width-128
  reduction dispatch.
- [x] Benchmark validates both device outputs against an exact host-side `u32`
  sum before reporting timings.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (29
  passed); `cargo test --doc --workspace --locked`; `cargo doc --workspace
  --no-deps --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  250,445 ns/iter; caller-owned 77,795 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 40,460 ns/iter; width-128 79,655 ns/iter for 65,536 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic benchmark
  validation, value-semantic tests, and empirical benchmark.

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

## 0.6.9 remaining invariant panic names [patch]
- [x] Replaced the unnamed `BlockWidth::DEFAULT` const panic with an explicit
  invariant message.
- [x] Normalized the strided bind-slot conversion `expect` message to the
  same `invariant:` convention as the other library panic sites.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; invariant-panic scan confirms every non-test panic
  site carries an `invariant:` message; `cargo nextest run --workspace
  --locked` (28 passed); `cargo test --doc --workspace --locked`; `cargo doc
  --workspace --no-deps --locked`; `cargo metadata --no-deps --locked
  --format-version 1`; `cargo bench --bench elementwise_into --locked` on
  real adapter (allocating 244,860 ns/iter; caller-owned 81,235 ns/iter for
  1,048,576 elements, 20 iterations); `git diff --check`. Evidence tier:
  source audit, value-semantic tests, and empirical benchmark.

## 0.6.8 library invariant panic messages [patch]
- [x] Replaced library-code unqualified `unwrap()` sites in reduction internal
  buffer selection, pipeline-cache locking, and transient-pool locking with
  explicit invariant `expect(...)` messages.
- [x] Confirmed remaining `unwrap()` sites in source scan are test-local.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; source `unwrap()` scan confirms remaining hits are
  test-local; `cargo nextest run --workspace --locked` (28 passed); `cargo
  test --doc --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 234,105
  ns/iter; caller-owned 79,575 ns/iter for 1,048,576 elements, 20 iterations);
  `git diff --check`. Evidence tier: source audit, value-semantic tests, and
  empirical benchmark.

## 0.6.7 value-semantic negative assertions [patch]
- [x] Replaced remaining broad absence and variant-only assertions in the
  audited Rust test scope with concrete mapped-value or length comparisons.
- [x] Confirmed no `is_err`, `is_ok`, `is_some`, `is_none`, or
  `assert!(matches!)` assertions remain under the audited source/test paths.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (28
  passed); assertion-pattern scan with `rg` over audited source/test paths
  returned no matches; `cargo test --doc --workspace --locked`; `cargo doc
  --workspace --no-deps --locked`; `cargo metadata --no-deps --locked
  --format-version 1`; `cargo bench --bench elementwise_into --locked` on
  real adapter (allocating 197,245 ns/iter; caller-owned 58,495 ns/iter for
  1,048,576 elements, 20 iterations); `git diff --check`. Evidence tier:
  value-semantic tests, assertion-pattern scan, and empirical benchmark.

## 0.6.6 negative-path contract assertions [patch]
- [x] Replaced remaining elementwise and strided negative-path `is_err()`
  assertions with typed `HephaestusError` checks.
- [x] Strided rejection tests now assert the zero-stride-output dispatch
  message and the exact layout storage error for backing-buffer overflow.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run -p hephaestus-wgpu
  strided_rejects_aliasing_output_and_short_buffers --locked` (1 passed);
  `cargo nextest run --workspace --locked` (28 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 160,095
  ns/iter; caller-owned 52,375 ns/iter for 1,048,576 elements, 20 iterations);
  `git diff --check`. Evidence tier: value-semantic contract tests and
  empirical benchmark.

## 0.6.5 contiguous elementwise alias guard [patch]
- [x] Added a shared output/input alias guard for caller-owned contiguous
  binary, unary, and scalar elementwise dispatch.
- [x] Added contract coverage for binary left/right aliases plus unary and
  scalar aliases, asserting the typed `DispatchFailed` error message.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (28
  passed); `cargo test --doc --workspace --locked`; `cargo doc --workspace
  --no-deps --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  215,105 ns/iter; caller-owned 84,150 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic contract
  tests and empirical benchmark.

## 0.6.4 transient pool best-fit reuse [patch]
- [x] Changed `BoundedBufferPool::take_at_least` to choose the smallest
  retained buffer that satisfies the requested size.
- [x] Added regression coverage preserving a large retained buffer after a
  small request consumes a smaller sufficient buffer.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (27 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 393,935
  ns/iter; caller-owned 85,240 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.3 transient pool FIFO storage [patch]
- [x] Replaced `BoundedBufferPool` backing storage with `VecDeque` so
  oldest-first count-cap eviction uses `pop_front()` instead of `Vec::remove(0)`.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (26 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 269,205
  ns/iter; caller-owned 90,290 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.2 adaptive transient pools [patch]
- [x] Changed `BoundedBufferPool::recycle` to evict oldest retained buffers
  when the count cap is full, then enforce the byte cap.
- [x] Added regression coverage for full-pool small-buffer pollution and a
  zero-count invariant.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (26 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 174,145
  ns/iter; caller-owned 95,335 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.1 bounded transient pools [patch]
- [x] Added `infrastructure::pool::BoundedBufferPool` with retained-buffer
  count and byte caps.
- [x] Routed staging and uniform pools through the bounded pool while keeping
  existing WGPU buffer reuse semantics.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`;
  `cargo fmt --check`; `cargo check --workspace --locked`; `cargo clippy
  --workspace --all-targets --locked -- -D warnings`; `cargo nextest run
  --workspace --locked` (24 passed); `cargo test --doc --workspace --locked`;
  `cargo doc --workspace --no-deps --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo bench --bench elementwise_into
  --locked` on real adapter (allocating 272,830 ns/iter; caller-owned
  132,710 ns/iter for 1,048,576 elements, 20 iterations); `git diff --check`.
  Evidence tier: type-level ownership plus value-semantic unit/contract tests
  and empirical benchmark.

## 0.6.0 caller-owned contiguous elementwise [minor]
- [x] Added `binary_elementwise_into`, `unary_elementwise_into`, and
  `scalar_elementwise_into` for caller-owned output buffers and `BlockWidth`
  selection.
- [x] Routed allocating contiguous elementwise APIs through the caller-owned
  implementations; scalar dispatch now uses the uniform pool.
- [x] Consolidated pipeline-cache creation into `application::pipeline` for
  contiguous elementwise, strided elementwise, and reduction kernels.
- Evidence: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (22 passed);
  `cargo test --doc --workspace --locked`; `cargo doc --workspace --no-deps
  --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter
  (allocating 291,410 ns/iter; caller-owned 66,350 ns/iter for 1,048,576
  elements, 20 iterations). Evidence tier: value-semantic differential tests
  and empirical benchmark, not a stored Criterion regression baseline.

## Default provider feature contract [patch]
- [x] Added default `parallel` and `mnemosyne-memory` feature markers to
  `hephaestus-core` and `hephaestus-wgpu`.
- Evidence: `cargo metadata --no-deps --locked --format-version 1`; full Atlas
  feature-policy metadata audit; `cargo check --workspace --offline`;
  `cargo test --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo doc --workspace --no-deps --locked`;
  `git diff --check`.

## 0.3.1 uniform pooling + CUDA ADR [patch]
- [x] Pooled strided meta uniforms (queue-ordered write_buffer reuse);
  one fewer buffer allocation per dispatch. 17 tests green on hardware.
- [x] ADR 0001 accepted (Phase 2 gate).

## 0.3.0 strided unary/scalar + consolidation [minor]
- [x] Shared strided core (SSOT): `StridedMeta` packing, WGSL Meta/decode
  fragments, `cached_pipeline`, `encode_strided` serve all strided kernels.
- [x] `unary_elementwise_strided_into` (broadcast + caller-owned output).
- [x] `scalar_elementwise_strided_into` — zero new kernels (one-element
  operand at all-zero strides through the binary kernel).
- [x] Tests: unary transposed sqrt, unary broadcast neg, scalar/binary
  equivalence over a transposed view; 17 total on real hardware.
- [x] Gates: fmt, clippy `-D warnings`, test, doc — clean.

## 0.2.0 strided dispatch [minor]
- [x] `binary_elementwise_strided_into` over leto `Layout<N>` (rank ≤ 4
  compile-time cap, leto broadcast semantics, caller-owned output,
  aliasing/short-buffer rejection, packed 80-byte Meta uniform).
- [x] Differential strided suite (5) on real hardware; 14 tests total.
- [x] Gates: fmt, clippy `-D warnings`, test, doc — clean.

Previous sprint (0.1.0 scaffold) below.
In-flight item: none. Next concrete increment: strided-layout-aware dispatch (backlog Phase 1).

## 0.1.0 scaffold [arch]
- [x] Workspace: `hephaestus-core` (no GPU deps, `#![forbid(unsafe_code)]`,
  `#![deny(missing_docs)]`) + `hephaestus-wgpu` (wgpu 26).
- [x] `ComputeDevice` seam with GAT `Buffer<T: Pod>`; `DeviceBuffer<T>`;
  distinct error variants (adapter/device/length/dispatch/transfer).
- [x] `WgpuDevice` acquisition (default + custom limits) — single authoritative
  copy of the logic formerly duplicated in apollo-wgpu-helpers.
- [x] `WgpuBuffer<T>`: PhantomData-typed, padded allocation, `raw()` escape
  hatch for consumer-owned pipelines.
- [x] Upload (`create_buffer_init`), zeroed alloc, download (staging +
  map_async + poll), length-mismatch rejection before transfer.
- [x] `binary_elementwise::<Op, T>`: ZST `BinaryWgslOp` markers (Add/Sub/Mul),
  `WgslScalar` type-token substitution, arrayLength tail guard, partial
  trailing workgroup correct.
- [x] `unary_elementwise::<Op, T>`: ZST `UnaryWgslOp` markers (Exp/Ln/Sin/Cos/Sqrt/Abs/Neg/Recip) and shared WGSL template.
- [x] `scalar_elementwise::<Op, T>`: uniform buffer binding and ZST-wrapped pipeline cache keys.
- [x] `reduction::<Op, T>`: ZST `ReductionWgslOp` markers (Sum/Min/Max), multi-pass tree reduction, and type-safe `ReductionIdentity` mapping.
- [x] Contract tests (9): round-trip values, length rejection (download + dispatch), add/mul/unary/scalar/reduction vs CPU reference.
  Verified on real adapter hardware; environment-gated skip otherwise.
- [x] Gates: `cargo fmt --check`, `clippy --all-targets -- -D warnings`,
  `cargo test`, `cargo doc --no-deps` — all clean.
- [x] Pushed to GitHub; apollo delegation integration (see backlog Phase 4).
