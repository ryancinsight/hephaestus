# Gap Audit - hephaestus

This register is the SSOT for hephaestus's known limitations. Items are grouped
by actionability so an open defect is never confused with an intentional
architectural decision or a tracked future-work item:

- **Resolved** — closed with evidence (kept for traceability).
- **Accepted design decisions** — intentional architecture, not defects.
- **Open future work** — tracked GPU-kernel / performance parity requiring
  `[major]` effort; the wrappers are API-complete and value-verified, the gap is
  native-kernel/performance parity, not correctness.
- **Environment / toolchain limitations** — blockers outside the source tree.

## [HEPH-FDTD-PROVIDER-1] Provider-owned 3D FDTD seam

- Finding: Hephaestus had no typed three-dimensional FDTD contract, leaving
  downstream consumers to own raw WGPU pipelines and making GPU/CPU
  equivalence unverifiable against the selected provider.
- Resolution: `hephaestus-core::domain::fdtd` now owns validated f32 grid
  parameters, medium and velocity storage layouts, and the `Fdtd3dOps` seam;
  `hephaestus-wgpu::application::fdtd` owns the prepared velocity-then-pressure
  kernels. Consumers retain source injection, medium population, CPU reference,
  and comparison policy.
- Evidence: local core/WGPU compilation, focused provider contract, and the
  independent one-step central-difference oracle pass. Device-required hosted
  WGPU CI and Kwavers cutover remain open.
- Residual: CUDA/ROCm implementations and multi-step consumer integration are
  separate dependent increments; no runtime or memory improvement is claimed.

## [HEPH-ATTENTION-CONFORMANCE-STRUCTURE-101] Attention contract file structure

- Finding: the provider attention contract exceeded the repository's 500-line
  implementation-file target after the merged attention work added a shared
  download assertion helper.
- Resolution: move the shared assertion into
  `crates/hephaestus-conformance/src/attention/assertions.rs`, leaving the
  attention contract manifest and backend-neutral logic below the target.
- Evidence: the local conformance scan reports `oversized_files=38` at source
  `702eba8`, down from 39. Exact-head CUDA `32026666522`, ROCm `32026666500`,
  WGPU `32026666544`, and Metal `32026666549` pass at merged default `4714b8c`.
  The direct Coeus cutover remains separately open under
  `HEPH-ATTENTION-PROVIDER-1`.

## [HEPH-CROSS-ENTROPY-PROVIDER-1] Accelerator cross-entropy ownership

- Finding: Hephaestus exposes no classification-loss role, forcing a downstream
  consumer to download accelerator logits and compute forward/backward on the
  host.
- Impact: selected backend identity does not match the executing provider;
  logits and gradients incur full-payload transfers and probabilities live in
  consumer-owned host memory.
- Resolution delivered: accepted ADR 0049 assigns one device-neutral mean
  cross-entropy seam to Hephaestus with provider-resident probabilities, one
  shared semantic-status and numerical-tolerance protocol, and vendor-only
  device implementations for WGPU, CUDA, ROCm, and Metal.
- Residual: no runtime or memory improvement is claimed until the complete
  consumer cutover has matched measurements. Exact-head hosted provider CI is
  closed at `bc6dfcf` by WGPU, CUDA, ROCm, Metal, and mdBook workflows
  (`31646386129`, `31646386243`, `31646386123`, `31646386192`,
  `31646386586`); the Coeus consumer cutover remains open.

## [HEPH-CUDA-COPY-SYNC-1] Device-local copy barrier

- Finding: CUDA `ComputeDevice::copy_buffer` issued `cuMemcpyDtoD_v2`, submitted
  its no-op host command stream, then called `cuCtxSynchronize`, imposing a
  context-wide completion barrier.
- Resolution: retain the canonical command-stream copy and submission while
  replacing context synchronization with `cuStreamSynchronize` on the default
  stream used by synchronous-form copies. NVIDIA documents that synchronous-form
  device-to-device copies issue through the default stream but perform no
  host-side synchronization:
  <https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html>.
- Evidence: an adapterless source contract pins one device-local copy, one
  submission, zero context synchronization calls, exactly one default-stream
  wait, exactly one `cuMemcpyDtoD_v2`, and zero `cuMemcpyDtoDAsync` calls in the
  underlying copy routine. Physical CUDA coverage value-checks a 1,027-element
  copy, zero-length copy, and typed length mismatch. Focused source and physical
  value runs each pass 1/1. The broader shared transfer conformance exposed a
  zero-sized POD defect: CUDA attempted `cuMemAlloc_v2(0)` for a nonzero logical
  element count. Allocation, upload, borrowed download, and writes now preserve
  the logical length while skipping every zero-byte driver call; corrected
  physical transfer conformance passes 2/2. Final adapterless source coverage
  passes 1/1 and formatting passes. Independent re-review approves with no
  remaining findings. Exact-final-diff Clippy/doctest collection timed out
  behind peer-held shared-target locks; hosted exact-head CI remains the
  warning/doc oracle. Exact implementation-head WGPU run `30774973252`, CUDA
  run `30774973245`, ROCm run `30774973255`, and native macOS Metal run
  `30774973240` pass.
- Residual: the change replaces one context-wide barrier with one default-stream
  wait by construction; nonzero transfer bytes, buffer ownership, and
  host-visible completion remain unchanged. No runtime speedup is claimed
  without matched hardware measurement.
  Hardware-only NVIDIA and AMD jobs skip because matching labeled runners are
  unavailable; PR #188's documentation-only closeout head remains the merge
  gate.

## [HEPH-WGPU-METAL-OWNED-READBACK-1] Mapped host result initialization

- Finding: WGPU and its Metal wrapper inherited the default
  `ComputeDevice::download_owned`, which zero-filled initialized host storage
  before mapped staging synchronously overwrote every result byte.
- Resolution: WGPU reserves host vector capacity, maps and validates the staging
  range through the same helper used by borrowed downloads, copies directly
  into spare capacity, and publishes length only after the helper succeeds.
  A mapping-lifecycle guard cancels pending maps or unmaps active ranges on
  every error and unwind exit before pooled staging storage is recycled. Metal
  delegates owned download explicitly to that implementation. Empty and
  zero-sized POD results perform no physical transfer.
- Evidence: the physical WGPU shared transfer contract passes with exact NaN
  payload, signed-zero, logical-length, empty no-allocation, and zero-sized POD
  clauses; a mapped-consumer unwind regression proves the staging allocation is
  reusable afterward. WGPU, Metal, and conformance compile warning-clean under
  `-D warnings`; independent re-review reports no remaining findings. Native
  Metal execution passes on native macOS in run `30735730960`; exact-head WGPU
  run `30735730194`, CUDA run `30735731839`, and ROCm run `30735732609` also
  pass. The source change removes one `O(n)` host initialization pass but makes
  no runtime or peak-memory improvement claim without controlled measurement.
- Residual: WGPU validation and physical execution are available locally;
  native Metal execution is unavailable on this Windows host but is covered by
  hosted macOS CI. Miri does not execute WGPU device mapping, so safety evidence
  combines the explicit initialized-byte invariant, value-semantic device
  execution, and independent review.

## [HEPH-WGPU-DECOMP-READBACK-1] Decomposition host initialization

- Finding: after WGPU gained provider-owned mapped readback, nine non-blocked
  decomposition families still allocated 16 initialized host vectors
  immediately before full mapped copies wholly overwrote them. Metal inherited
  the same path through its WGPU-backed device.
- Resolution: route matrix inputs and solve right-hand sides in bidiagonal,
  Bunch-Kaufman, column-pivoted QR, eigenvalue, complete-pivoted LU, Hessenberg,
  Schur, SVD, and UDU families through `ComputeDevice::download_owned`. Leave
  blocked LU, QR, and Cholesky untouched under the active KS-5 claim.
- Evidence: a syntax-aware source regression rejects direct caller-owned heap
  readback, including method and UFCS forms, and pins the exact 16 owned calls
  across every claimed module. The shared decomposition conformance suite
  preserves factorization, solve, reconstruction, invalid, and empty values.
  Both focused tests pass under the exact committed Git graph. Formatting,
  all-target warning-denied Clippy, doctests, and warning-clean Rustdoc pass;
  independent re-review reports no remaining finding. Exact implementation
  head `d27cfd6` passes WGPU run `30760402397`, CUDA run `30760402388`, ROCm run
  `30760402390`, and native macOS Metal run `30760402389`. Hardware-only NVIDIA
  and AMD jobs skip because this dispatch did not request self-hosted devices.
- Residual: allocation count, transfer volume, device storage, and peak host
  storage are unchanged. The source change removes 16 linear host
  initialization writes by construction; no runtime or peak-memory improvement
  is claimed without controlled hardware measurement. Native Metal execution
  remains hosted macOS evidence.

## [HEPH-METAL-ACQUISITION-1] Backend-neutral Metal device acquisition

- Finding: WGPU, CUDA, and ROCm implemented `ComputeDeviceAcquisition`, while
  Metal exposed only its backend-specific `try_default` constructor. Generic
  consumers could query Metal capabilities but could not acquire it through the
  shared provider seam.
- Resolution: implement the shared seam by extending the WGPU substrate with
  Metal-only single and bounded enumeration paths. Single acquisition maps the
  shared power preference onto native adapter selection; enumeration ranks
  discrete and integrated adapters by the same preference. Both paths intersect
  optional features with adapter support and enforce the existing typed limit
  translation. A zero-device bound returns before instance or adapter probing.
- Evidence: focused contracts pin the zero bound, Metal backend identity,
  optional feature mapping, exact requested limits, and one-device bound. The
  WGPU policy unit test pins both preference orderings. Exact-source focused
  Nextest passes 3/3 Metal contracts and the 1/1 WGPU policy unit; WGPU and
  Metal all-target compilation, warning-denied Clippy, doctests, warning-clean
  Rustdoc, and formatting pass. WGPU SemVer passes 196/196 applicable checks;
  Metal SemVer is blocked before API analysis by an unexpected MSVC linker
  failure in the tool's temporary rustdoc graph. Independent re-review reports
  no remaining finding. Exact implementation head `3b8cc85` passes WGPU run
  `30762519498`, CUDA run `30762519504`, ROCm run `30762519499`, and native
  macOS Metal run `30762519503`. Hardware-only NVIDIA and AMD jobs skip because
  this dispatch did not request self-hosted devices.
- Residual: no Metal adapter is available on this Windows host. The change is
  capability parity, not a runtime or memory optimization; no performance claim
  is made.

## [HEPH-WGPU-METAL-OWNED-READBACK-1] Themis package identity blocker

- Finding: exact-head Metal and ROCm CI failed before compilation because
  upstream Themis renamed its Cargo package from `themis` to
  `themis-topology` at `a1c8231`; Hephaestus still requested the old package
  identity from the Git repository.
- Resolution: keep the source-level `themis` crate alias and bind it explicitly
  to `package = "themis-topology"`, version `0.10.1`. Refresh the generated
  stack overlay and Hephaestus lockfile against the renamed package.
- Evidence: fresh hosted resolution reaches and passes provider compilation and
  contracts at exact head `1c4ac16`: CUDA run `30735731839`, ROCm run
  `30735732609`, WGPU run `30735730194`, and native macOS Metal run
  `30735730960`. The earlier failed Metal run `30733059974` and ROCm run
  `30733059964` remain resolver evidence only, not backend evidence.
- Follow-up: Mnemosyne subsequently renamed packages `mnemosyne` and
  `mnemosyne-core` to `mnemosyne-memory` and `mnemosyne-memory-core` at
  `be4aa64`. Direct consumers retain their Rust crate aliases, bind the new
  Cargo identities, and pass the exact-head provider matrix recorded above.

## [HEPH-OWNED-DOWNLOAD-1] Host result initialization

- Finding: CUDA and ROCm pseudoinverse and matrix exponential each allocated a
  zero-filled host vector immediately before a synchronous device-to-host copy
  overwrote every byte. The caller-owned `ComputeDevice::download` contract
  could not express provider allocation without duplicating that initialization.
- Resolution: add a default-compatible `download_owned` provider method.
  CUDA and ROCm reserve fallible host capacity, copy synchronously into spare
  capacity, and publish vector length only after success; zero-sized types use
  safe initialized resize. Four matrix-function consumers use the owned result
  directly. WGPU and Metal retained the initialized default in that increment;
  HEPH-WGPU-METAL-OWNED-READBACK-1 closes that residual.
- Evidence: shared bitwise transfer conformance includes NaN payloads, signed
  zero, logical length, and empty no-allocation behavior. WGPU conformance passes
  1/1; physical CUDA conformance plus matrix functions pass 4/4; adapterless
  ROCm matrix-function contracts pass 3/3. Feature-enabled and feature-off
  warning-denied checks pass. Independent review approves the copy and public
  API contracts after requiring the safe ZST branch and changelog entry.
- Residual: no physical ROCm device is available on this Windows host; the HIP
  path requires hosted ROCm CI. Miri cannot execute CUDA/HIP driver calls. The
  source change removes four `O(n)` host initialization writes but does not
  change allocation count or peak memory; no runtime gain is claimed without a
  controlled hardware benchmark. `hephaestus-core` is unpublished, so
  `cargo-semver-checks` has no registry baseline. Implementation head
  `dad12d6` passes CUDA run `30713948491`, ROCm run `30713948448`, WGPU run
  `30713948446`, and macOS Metal run `30713948456`; hardware-only NVIDIA and AMD
  jobs skip because no runner was dispatched. The external
  `recurseml/analysis` generic error is not backend evidence. Final docs-only
  exact-head CI remains PR #175's merge gate.

## [HEPH-PREPARED-L2-OVERWRITE-1] Prepared output lifecycle

- Finding: the earlier fused map-reduction changelog overclaimed that prepared
  CUDA/ROCm L2 results omit scalar initialization. Although successful
  square-root dispatch overwrites the sole element, `output()` exposes the
  allocation before first dispatch and remains readable after a failed
  dispatch. Raw uninitialized CUDA/ROCm storage would therefore violate the
  public lifecycle.
- Resolution: retain defined zeroed storage in all four public-strided and
  internal-dense preparation paths. Poison the stable output with `NaN` before
  first and repeated successful dispatches so the value contract proves kernel
  assignment without using that proof to weaken pre-dispatch safety. Correct
  the changelog to limit overwrite allocation to immediate L2 results.
- Evidence: physical CUDA passes the focused prepared-map contract before and
  after the allocation change; adapterless ROCm compiles and exercises the
  typed unavailable-device path. Formatting, warning-denied provider Clippy,
  and no-default all-target checks pass locally.
- Residual: local physical ROCm execution is unavailable on this Windows host;
  no physical ROCm execution is claimed. Implementation head `998f521` passes
  CUDA run `30710893281`, ROCm run `30710893299`, WGPU run `30710893294`, and
  macOS Metal run `30710893263`; hardware-only NVIDIA and AMD jobs skip because
  no runner was dispatched. The external `recurseml/analysis` integration
  reports its generic service error and is not provider evidence. Final
  docs-only closeout-head CI remains PR #173's merge gate. No allocation,
  peak-memory, transfer, or runtime improvement is claimed.

## [HEPH-LGAMMA-EXPRESSION-PARITY-1] Log-gamma vocabulary

- Finding: Leto and Coeus expose `lgamma`, but Hephaestus had no shared marker;
  Coeus WGPU explicitly rejected it.
- Resolution: add `LgammaOp`; CUDA and HIP use native `lgamma`, while WGPU and
  Metal use the provider-owned Lanczos/reflection expression.
- Residual: f64/reduced/vector contracts and digamma gradients remain open;
  the f32 provider and Coeus consumer paths are complete.
- Evidence: core expression tests cover native forms, Lanczos coefficients, and
  pole/infinity selection. Hephaestus PR #118 passed WGPU `90086428952`, CUDA
  `90086430178`, ROCm `90086430143`, and Metal `90086428160`. Coeus PR #231
  merged at `971fab9614b97bd708a716d01684da58fd1331ba`; its consumer jobs
  passed WGPU `90088836682`, CUDA `90088836688`, ROCm `90088836731`, and Metal
  `90088836675`. Required-device ROCm job `90088837591` was skipped; no
  physical-device execution claim is made.
- Status: resolved for f32 provider and consumer parity; extended scalar/vector
  contracts and digamma remain future work.

## [HEPH-ERROR-FUNCTION-EXPRESSION-PARITY-1] Error-function vocabulary

- Finding: Leto and Coeus expose f32 `erf` and `erfc`, but the shared
  Hephaestus unary marker seam did not expose either operation to ROCm or
  Metal.
- Resolution: add `ErfOp` and `ErfcOp`; WGPU uses the existing
  Abramowitz–Stegun expression, CUDA and HIP use native intrinsics, and Metal
  delegates through WGPU.
- Residual: broader non-f32 contracts and unrelated expression families remain
  open; the f32 provider and Coeus consumer paths are complete.
- Evidence: provider docs head `df8a896` passed WGPU `90028947591`, CUDA
  `90028946846`, ROCm `90028946770`, and Metal `90028947450`. Coeus PR #228
  merged at `aca9a5a8`; final docs head `08614299` passed run `30283857017`
  with CUDA `90036655765`, ROCm `90036655656`, Metal `90036655618`, and WGPU
  `90036655846`. Required hardware jobs skipped because no physical device
  runner was selected.
- Status: resolved. Provider implementation and Coeus consumer routing are
  complete.
- Evidence: provider docs head `df8a896` passed WGPU job `90028947591`, CUDA
  `90028946846`, ROCm `90028946770`, and Metal `90028947450`. Coeus PR #228
  merged at `aca9a5a8`; final docs head `08614299` passed run `30283857017`
  with CUDA `90036655765`, ROCm `90036655656`, Metal `90036655618`, and WGPU
  `90036655846`. Required hardware jobs skipped because no physical device
  runner was selected.

## [HEPH-UNARY-MATH-EXPRESSION-PARITY-1] Unparameterized unary math vocabulary

- Finding: Leto and Coeus define a broader f32 unary math vocabulary than the
  Hephaestus expression markers consumed by the four accelerator providers.
- Resolution: add one shared dialect-specific marker per operation and export
  it through WGPU, CUDA, ROCm, and Metal. The current implementation covers
  tangent, inverse and hyperbolic functions, logarithm/exponential bases,
  `expm1`, `log1p`, sign, and rounding.
- Residual: parameterized activations and f64/vector contracts remain outside
  the current common baseline; the Lgamma, GELU, and error-function increments
  are tracked by their resolved parity records.
- Evidence target: core expression tests, Coeus ROCm/Metal Leto differential
  tests, and exact-head WGPU, CUDA, ROCm, and Metal workflows. Physical-device
  runner execution is reported separately from adapterless provider CI.
- Status: Hephaestus implementation and exact-head provider evidence are
  complete at `b088a2f`; WGPU `89997918070`, CUDA `89997916944`, ROCm
  `89997920644`, and Metal `89997917574` passed. Coeus consumer routing and
  differential verification remain open in the downstream repository.

## [HEPH-SCAN-SUFFIX-PARITY-1] Reverse cumulative-sum convenience surface

- Finding: Leto exposes reverse cumulative sum, but the four Hephaestus roots
  exposed only the generic reverse scan and reverse cumulative product helpers.
- Resolution: add one provider-owned `suffix_sum`/`suffix_sum_into` wrapper over
  the existing `CumSumOp` reverse scan. Metal delegates through its WGPU
  substrate; no duplicate kernel body is introduced.
- Evidence target: each provider contract compares both output forms with
  `leto_ops::scan_axis::<CumSumOp, _, 2>(..., ScanDirection::Reverse)` and the
  hosted WGPU, CUDA, ROCm, and Metal workflows pass on one commit head.
- Status: implementation complete; local compilation remains blocked by the
  checkout-local Leto/Hermes provider graph, so no local build claim is made.

## [HEPH-LAPLACIAN-CONTRACT-1] Shared typed stencil (2026-07-20)

- Finding: the WGPU provider duplicated boundary codes and host validation,
  while its differential test carried a second CPU stencil implementation.
- Resolution: derive parameters from Leto's `Laplacian2D`, re-export the Leto
  boundary and polarity types, and use Leto Ops as the CPU differential oracle.
- Evidence tier: compile-time type unification, exact signed-coefficient unit
  coverage, real-device differential tests when an adapter is available, and
  focused package gates (configured Nextest 152/152; doctest and
  warning-denied rustdoc pass).
- Residual: WGSL remains intentionally `f32`; Leto CPU execution also covers
  `f64`, whose accelerator support requires a backend with native storage.

## [HEPH-CUDA-FEATURE-HYGIENE] CUDA-only warning closure (2026-07-20)

- Finding: `PinnedHostBuffer` and three decomposition pipeline keys compiled
  whenever CUDA was enabled, although their only consumers require the
  independent `decomposition` feature.
- Resolution: gate those private implementation details on the conjunction of
  `cuda` and `decomposition`.
- Evidence tier: feature-matrix static verification plus configured native
  tests; warning-denied all-target Clippy passes for both feature combinations,
  and Nextest passes 109/109.
- Residual: none; public APIs and runtime behavior are unchanged.

## [HEPH-EUNOMIA-0.4-REFRESH] Provider lock (2026-07-18)

- Resolution: advance the lock from Eunomia 0.2.0 `34d0cc8a` to 0.4.0
  `49dc115e`, carrying the canonical sub-byte conversion kernel and corrected
  reduced-format constants into every Hephaestus backend.
- Evidence tier: dependency-resolution identity plus warning-denied
  all-target/all-feature Clippy, configured Nextest 312/312, doctest, and
  warning-denied rustdoc.
- Residual: the refresh changes no Hephaestus source or public API.

## [HEPH-EUNOMIA-0.6-REFRESH] Native reduced-precision closure (2026-07-19)

- Finding: Hephaestus's lock selected Hermes 0.3 and Leto 0.38, whose
  `half::f16`/`half::bf16` SIMD contracts required Eunomia's deleted foreign
  numeric implementations.
- Resolution: advance the coherent provider closure to Eunomia 0.6.0
  `df77dfde`, Hermes 0.4.0 `c9bbdf8a`, and Leto 0.39.0 `7afcbd0e`.
- Evidence tier: dependency-resolution identity; all-target/all-feature
  workspace check; warning-denied Clippy; real-device and CPU Nextest 312/312;
  doctests; warning-denied rustdoc.
- Residual: the lock refresh changes no Hephaestus source or public API.

## [HEPH-EUNOMIA-COMPLEX-1] Eunomia complex ownership (2026-07-18)

- Finding: general-eigenvalue APIs owned `num_complex::Complex<f32>` even
  though Leto already returned `eunomia::Complex<f32>`. WGPU/CUDA rebuilt the
  same representation field-by-field, and the Python boundary allocated a
  second vector solely to change the Rust type identity.
- Resolution: WGPU, CUDA, Metal, and Python use Eunomia's type directly; Leto
  eigenvalue vectors upload without conversion; NumPy construction consumes
  the downloaded Eunomia vector directly through Eunomia's `numpy::Element`
  contract. Direct `num-complex` manifests and source references are deleted.
- Evidence tier: compile-time type unification and committed dependency lock;
  WGPU/CUDA real-device value-semantic eigenvalue contracts; Python NumPy
  differential parity; 264/264 affected Nextest cases; warning-denied Clippy
  and rustdoc; doctests; supported minimal-feature checks; source/manifest
  residue scan; and public API SemVer analysis.
- Residual: `num-complex` remains transitively inside NumPy/ndarray, an
  external FFI implementation detail that does not enter Hephaestus source or
  public buffer types.

## [HEPH-LEGACY-MATH-RESIDUE-1] Leto-only CPU references (2026-07-17)

- Finding: the provider still carried direct `ndarray`/`nalgebra` edges only
  for comparative benchmark baselines and WGPU differential oracles, creating
  a second CPU vocabulary beside the Atlas array/linalg provider.
- Resolution: remove those manifest edges, replace the differential oracles
  with Leto/Leto Ops, and reduce both comparative benches to real Leto-versus-
  provider measurements for elementwise, reduction, and matrix products.
- Theorem: for a fixed input (x), each comparison now evaluates the same
  operation (f) through exactly two implementations, `leto_ops::f(x)` and
  the provider dispatch (P_f(x)); the downloaded provider result is checked
  against the Leto storage oracle before either timing loop. No third-party
  reference can define a competing shape, layout, or tolerance contract.
- Evidence tier: compiler-checked dependency removal, 48/48 core tests,
  140/140 WGPU tests, 109/109 CUDA tests, warning-denied Clippy, doctests,
  warning-clean rustdoc, and all-target benchmark compilation. The Python
  `numpy` bridge remains an external FFI representation and is not a domain
  compute dependency.

## [HEPH-SCAN-LIMIT-AUDIT] Scan line-length bound (2026-07-17)

- Finding: KS-5b proposed a multi-pass block-sums/uniform-add extension on the
  premise that long lines exceed the current workgroup/shared-memory limit.
- Resolution: the current provider path assigns `W` lanes to one line; each
  lane folds `ceil(L/W)` values and stores one total, then the ordered prefix
  pass applies one preceding value per lane. Shared storage is exactly `W`
  partials, so its size is independent of line length `L`.
- Theorem: for `L >= 1`, `shared_bytes = W * size_of(T)` and
  `work_per_lane <= ceil(L/W)`. Therefore `L > W` does not require a
  multi-pass dispatch; it changes loop count only. The WGPU and CUDA
  `L = 513`, `W = 256` integer contracts are the value-semantic witness.
- Evidence tier: source algebra and existing real-device contract tests. No
  correctness defect remains. Re-open only after a measured device-specific
  workgroup/latency limit and a derived floating-point bound for reordered
  accumulation are recorded.

## Open feature hygiene

No open feature-combination defect is currently recorded in the backend scope.

## Open Future Work — GPU-kernel & performance parity

These surfaces are API-complete and value-verified against Leto/nalgebra; the
remaining gaps are native-GPU-kernel and/or performance parity (`[major]`
effort), not correctness. Factorization/solve currently delegate to Leto on the
host before uploading device buffers.

- [patch] CUDA ZST/empty upload defect (surfaced by KS-5 verification,
  2026-08-02): `cuda_satisfies_the_transfer_contract` (conformance clause 001m)
  fails on this Windows driver because a zero-byte `upload` reaches
  `cuMemAlloc_v2(0 bytes) -> 1` (CUDA_ERROR_INVALID_VALUE) in the CUDA device
  allocation path. The empty `lu_decompose_blocked`/`lu_decompose` paths avoid
  it, but the general upload/allocation path must handle the zero-byte case
  (allocate a zero-size handle or skip allocation, matching the empty-upload
  contracts other providers satisfy). Pre-existing on origin/master (no
  infrastructure/transfer code differs between this branch and origin/master);
  filed here rather than absorbed into the KS-5 LU slice. Evidence tier:
  `cargo nextest run -p hephaestus-cuda --all-features` — 143/144 pass, this
  one fails with `ZST upload: AllocationFailed { cuMemAlloc_v2(0 bytes) -> 1 }`.

- [arch] CUDA multi-storage beamforming dispatch is still a future concrete
  provider implementation. The generic trait and WGPU implementation are
  delivered; adding CUDA requires a real CUDA beamforming kernel and launch path,
  then downstream Kwavers verification against that provider. No Kwavers helper
  layer is required.
- [patch] WGPU axis reductions still carry fixed dispatch/synchronization
  overhead against CPU backends on small workloads after the short-axis
  workgroup reduction path, axis-0 tiling, prepared dispatch, batched axis
  submission, mixed scalar/axis submission, scalar final-pass collapse, and
  Leto's row-major rank-2 axis-0 CPU fast path. Mixed scalar/axis batches now
  remove one command encoder and queue submission without adding scratch
  buffers. The measured 64-column axis-0 tile halves default-width workgroups
  and lowers aggregate tree barriers from 32 to 12 for 256x256 inputs; local
  three-sample medians improve from 55.522 to 39.212 microseconds for one
  reduction and from 28.494 to 25.286 microseconds for eight. Current residual:
  scalar sum beats `ndarray`, but Leto CPU axis reductions remain faster than
  WGPU for 256x256 axis 0.
  Definition of ready for the next reduction slice: prototype a measured
  small-axis routing policy or fuse multiple axis statistics into one WGPU pass;
  do not target Hermes SIMD arithmetic until a CPU profile shows the arithmetic
  loop rather than layout/launch overhead is dominant. Evidence tier:
  value-semantic contract plus empirical comparative benchmark.
- [minor] WGPU Cholesky/LU/QR provide device-resident factors and Leto-matching
  solve/inverse/determinant surfaces, but factorization delegates to Leto on the
  host before uploading the factors (API parity, not GPU-kernel parity). Evidence
  tier: implementation audit, value-semantic differential tests, and comparative
  benchmark rows.
- [minor] WGPU symmetric Jacobi eigen decomposition provides device-resident
  eigenvalues/eigenvectors, but the eigensolve delegates to Leto on the host
  before uploading the outputs (API parity, not GPU-kernel eigensolver parity).
  Evidence tier: value-semantic differential tests, non-symmetric rejection test,
  and comparative benchmark row.
- [minor] WGPU general eigenvalues are exported with complex device buffers and
  covered for diagonal, exact complex-pair blocks, triangular, structured
  nonsymmetric real-spectrum, dense `nalgebra` differential, symmetric-real, and
  rectangular-rejection cases (32x32 block-rotation benchmark against Leto and
  `nalgebra`). Remaining risk is API/performance parity only: the wrapper
  delegates to Leto on the host before uploading complex device buffers. Evidence
  tier: value-semantic closed-form, differential, invalid-input tests, and
  empirical benchmark row.
- [minor] WGPU pseudoinverse and matrix exponential have non-diagonal,
  rank-deficient, rectangular, nilpotent, skew-symmetric, general-matrix, and
  invalid-input contract coverage plus comparative benchmark rows. Remaining risk
  is performance/API parity only: both wrappers delegate to Leto on the host and
  upload device buffers. Evidence tier: value-semantic closed-form, Moore-Penrose
  algebraic, differential, invalid-input tests, and empirical benchmark rows.
- [minor] WGPU blocked Cholesky offloads the trailing SYRK update to a GPU
  kernel, but diagonal panel factorization and triangular panel solves remain
  CPU/Leto-backed. Current empirical row: 128x128 blocked Cholesky is slower than
  Leto and `nalgebra` on the local WGPU run. Evidence tier: value-semantic
  differential test across a block boundary and empirical benchmark row in
  `benchmark_results.md`.
- [minor] WGPU blocked LU and blocked QR have comparative benchmark rows. Blocked
  LU transfers are narrowed to the active diagonal-panel and trailing-submatrix
  regions. Blocked QR transfers compact trailing-column tiles per panel before
  GPU Householder application and uploads all panel Householder vectors in one
  packed buffer. The measured 66x66 blocked LU row remains slower than Leto and
  `nalgebra`; the 70x35 blocked QR row is much slower than Leto and `nalgebra`.
  Evidence tier: value-semantic blocked LU/QR tests plus empirical benchmark rows.
- [minor] Blocked decomposition synchronization profiling shows a material, noisy
  transfer/synchronization floor after the blocked LU region-transfer reduction,
  blocked QR compact-tile transfer reduction, and packed reflector upload.
  Timestamp queries measure the QR launch component directly: 32 separate
  reflector-equivalent compute passes previously totaled 155.2 µs on the local GPU
  timeline (3.4 µs median pass). The WGPU QR panel path now applies all panel
  reflectors in one compute pass per panel; the timestamp profile is 8.2 µs total
  (160 ns median), and the 70x35 blocked QR row measures 420.8 µs. The production
  path constructs the host-side `QrDecomposition` from blocked factors with
  `from_raw_parts`; the obsolete final-Leto-recompute profile row is removed.
  The component profile measures the 70x35 CPU panel-factorization lower bound
  at 26.3 µs, while the synthetic QR host/device synchronization floor remains
  222.6 µs. The trailing-update kernel
  packs Householder vector offsets and beta coefficients into one reflector
  metadata buffer (two storage bindings → one). The 70x35 comparative row did not
  improve after this packing change: WGPU 480.8 µs vs Leto 14.9 µs and `nalgebra`
  10.0 µs. The retained profile now executes the production QR path instead of
  the superseded synthetic transfer schedule. The two-panel regime uses one
  dense download and one `R` upload; wider matrices retain blocked GPU
  Householder updates. Evidence tier: value-semantic blocked QR tests,
  production-executing component profiles, comparative benchmarks, and the
  historical GPU-timeline measurements in `benchmark_results.md`.
- [minor] WGPU CSR sparse storage uploads Leto CSR matrices into device-resident
  values plus one packed index buffer and executes SpMV/SpMM in WGSL without
  downloading operands to the host. The kernel layout stays within WGPU's portable
  four-storage-buffer limit, and dispatch sizing reuses the shared Mnemosyne/Moirai
  launch-planning helper. The focused sparse comparative harness validates WGPU
  outputs against Leto before timing and now times reusable caller-owned outputs:
  latest prepared SpMV 1000x1000 CSR measured WGPU 61.146 µs vs Leto 1.232 µs,
  latest `spmv_many` measured WGPU 62.758 µs vs repeated Leto SpMV 150.414 µs,
  and latest warmed batched prepared SpMM 1000x1000x128 measured WGPU 12.258 µs
  vs Leto 35.232 µs with the dense RHS fast path.
  Remaining risk: sparse performance parity is not achieved for either SpMV or
  SpMM on this run; no `ndarray`/`nalgebra` sparse comparator is recorded because
  the current Leto sparse API benchmark has no dense-library sparse equivalent in
  this harness. Evidence tier: static diagnostics, value-semantic WGPU sparse
  contract test, value-checked benchmark outputs, and empirical benchmark.

## Environment / Toolchain Limitations

- [patch] CUDA-enabled build generation requires both
  `LIBCLANG_PATH=D:\\msys64\\mingw64\\bin` and
  `PATH=D:\\msys64\\mingw64\\bin;%PATH%` on this host. Bindgen rejects the UCRT
  `libclang.dll`, while that MinGW LLVM environment builds
  `hephaestus-cuda --all-targets --locked` and the formerly blocked
  `hephaestus-core`/`hephaestus-wgpu --all-targets --all-features --locked`.
  Evidence tier: compile-time validation; this does not establish CUDA device
  execution parity.

- [patch] The 0.12.0 to 0.13.0 semver classification completes for
  `hephaestus-core`, `hephaestus-metal`, and `hephaestus-wgpu` as a pre-1.0 major
  change. `hephaestus-cuda` and `hephaestus-python` rustdoc generation is blocked
  because cargo-semver-checks builds current and baseline dependency graphs into
  one target tree, causing the GNU `cc` probe for `psm`/`stacker` to fail while
  the ordinary workspace rustdoc gate passes. Re-run those two packages when
  cargo-semver-checks isolates build outputs. Evidence tier: tool diagnostics
  (2026-07-13).
- [minor] CUDA mirrors the current core operation and decomposition slice in the
  source tree and passes stub-mode verification. Real CUDA feature verification is
  still required on CUDA hardware/toolchain before claiming device-execution
  parity for the CUDA kernels. CUDA blocked Cholesky remains CUDA-feature gated and
  is not part of the default stub-mode claim. Evidence tier: static diagnostics and
  stub-mode contract tests. (Blocked on CUDA hardware availability.)
- Native CUDA ABI/loading and physical-device verification are tracked in
  [HEPH-CUDA-DRIVER-BOUNDARY](backlog.md#heph-cuda-driver-boundary).

## Next Increment

- Continue KS-5b with a bounded multi-pass block-sums/uniform-add extension for
  lines that exceed provider workgroup/shared-memory limits. Preserve the
  generic `CombineExpr`/`IdentityToken` seam and derive the floating-point
  differential bound before accepting reordered results; otherwise continue
  the measured scalar-reduction launch-overhead slice.
