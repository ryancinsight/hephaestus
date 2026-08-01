# Backlog — hephaestus

Strategic roadmap; tags `[patch]`/`[minor]`/`[major]`/`[arch]` per SemVer class.
Source decision: atlas ADR 0001 (shared GPU substrate; wgpu + CUDA composing
cuda-oxide + cutile).

## HEPH-METAL-VOLUME-OVERWRITE-1 [patch] [perf] — review

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
- Status: implementation, independent review, local static gates, and hosted
  backend CI complete 2026-07-31; PR #168 is ready to merge.

## HEPH-ATTENTION-PROVIDER-1 [minor] [arch] — in-progress

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
- Status: in-progress 2026-07-31.
- Provider evidence (2026-07-31): WGPU and physical CUDA execute the shared
  semantic conformance suite; CUDA additionally verifies native `f64` additive
  backward. ROCm passes its Windows no-default-feature source/static contract;
  native HIP execution remains a hosted Linux gate. Every provider resets one
  device status word per prepared dispatch, validates in parallel before any
  caller-visible mutation, and reads back only that status word.

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
