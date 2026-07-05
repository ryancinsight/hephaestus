# Changelog

SemVer 2.0.0; pre-1.0 minor bumps may include breaking changes (documented).

## Unreleased

### Changed

- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` [minor]: completed
  KS-5 reduction planner parity. Axis-reduction metadata packing,
  shape/stride/output/alias validation, scalar reduction width validation, and
  scalar pass-depth planning now live in `hephaestus_core::reduction`; WGPU and
  CUDA keep only dialect shader generation, buffer ownership, and launch
  mechanics. Evidence: `cargo fmt -p hephaestus-core -p hephaestus-wgpu -p
  hephaestus-cuda --check`, `cargo check -p hephaestus-core`, `cargo check -p
  hephaestus-cuda --no-default-features`, `cargo check -p hephaestus-wgpu`,
  `cargo check -p hephaestus-cuda`, `cargo nextest run -p hephaestus-core
  reduction` (6/6), `cargo nextest run -p hephaestus-cuda --no-default-features
  reduction` (4/4), `cargo nextest run -p hephaestus-cuda reduction` (4/4),
  `cargo nextest run -p hephaestus-wgpu reduction` (5/5), and `cargo clippy -p
  hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda --all-targets
  --no-deps -- -D warnings`.
- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` [minor]: hoisted the
  duplicated axis-scan host orchestration (`ScanDirection`, `AxisScanMeta`,
  validation, metadata packing, workgroup count) into
  `hephaestus_core::scan::plan_axis_scan`, a backend-neutral SSOT. Both backends
  call it and keep only their dialect shader and raw dispatch (net -212 lines
  across the two scan modules). Adds a std-only `leto` dependency to
  `hephaestus-core` (ADR-0001's backend-agnostic layout vocabulary; core stays
  GPU-free). Behavior preserved — the differential-vs-leto scan tests pass
  unchanged (wgpu 129/129, cuda 102/103).

- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` [minor]: hoisted the
  decomposition operand validators (`validate_square`, dense-C-contiguous
  operand check) into `hephaestus_core::domain::decomposition`
  (`validate_square_operand`, `require_dense_operand`); each backend's
  `decomposition/validate.rs` becomes a thin adapter. Removes the last
  duplicated copy of the dense-operand OOB guard body. All 88 decomposition
  tests (incl. the adversarial dense-operand cases) pass unchanged.

### Fixed

- `hephaestus-cuda` / `hephaestus-wgpu` [patch]: restored all-targets
  compilation after Leto general-eigenvalue APIs returned `leto::Complex<f32>`
  while Hephaestus device buffers and comparative benchmarks use
  `num_complex::Complex<f32>`. CUDA eigenvalue upload and WGPU comparative
  reference checks now convert through explicit value-semantic real/imaginary
  field mapping.

- `hephaestus-cuda` [patch]: reconciled the Stage 1 CUDA substrate with ADR
  0001. Device acquisition, context binding, device allocation, typed
  `CudaBuffer<T>` ownership, upload/download/subrange transfers, kernel module
  unload, and buffer release now route through cuda-oxide driver bindings.
  Buffers use `CUdeviceptr` with `PhantomData<T>` and retain their cuda-oxide
  context so destruction binds the owning context before `cuMemFree_v2`. This
  replaces the previous managed-memory allocation path. CUDA allocation hints
  now resolve to one explicit non-managed primary-buffer tier: allocatable
  hints record `MemoryTier::Device`, budget-only tiers are rejected, and
  `MappablePrimaryBuffers` is false. This resolves the former WDDM
  `STATUS_IN_PAGE_ERROR` residual in
  `concurrent_device_acquisition_is_safe`. The blocked-decomposition region
  helper uses row-wise 1D copies instead of cuda-oxide 0.4.0's
  Windows-incompatible `CUDA_MEMCPY2D` layout. Evidence: `cargo fmt -p
  hephaestus-cuda --check`, `cargo check -p hephaestus-cuda`, `cargo check -p
  hephaestus-cuda --no-default-features`, both default and no-default
  `cargo clippy -p hephaestus-cuda --all-targets --no-deps -- -D warnings`,
  `cargo nextest run -p hephaestus-cuda` passes 105/105 on live CUDA, `cargo
  nextest run -p hephaestus-cuda --no-default-features` passes 60/60 via
  skip-without-driver contracts, `cargo test --doc -p hephaestus-cuda` passes
  0 doctests, and `cargo doc -p hephaestus-cuda --no-deps` passes. Build note:
  cuda-oxide 0.4.0's build script links `cuda.lib`, so the repo config supplies
  `CUDA_LIB_PATH` for the default CUDA feature. Current focused closure checks:
  `cargo nextest run -p hephaestus-cuda concurrent_device_acquisition_is_safe`
  (1/1), `cargo nextest run -p hephaestus-cuda
  device_capabilities_are_driver_backed` (1/1), and `cargo nextest run -p
  hephaestus-cuda test_placement_aware_allocation` (1/1).

- `hephaestus-cuda` [patch]: resolved the WDDM `STATUS_IN_PAGE_ERROR`
  (`0xc0000006`) managed-memory kernel-launch aborts on Windows.
  Root-caused by experiment (correcting the earlier placement-advice
  hypothesis): WDDM does not support concurrent host/device access to
  `cuMemAllocManaged` ranges, so a host allocation issued while a kernel is in
  flight on the null stream — the next intermediate buffer in multi-pass
  reductions and map-then-reduce (dot/norm/trace) — faults. A Windows-gated
  `cuCtxSynchronize` after each `cuLaunchKernel` drains the context before the
  next host managed-memory access. The backend is already null-stream-serial,
  so throughput is unchanged and this also attributes async kernel faults to
  the launching operation; Linux/UVM keeps launches asynchronous. The remaining
  managed-path residual was closed by the cuda-oxide `cuMemAlloc_v2` Stage 1
  substrate reconciliation above. Evidence: the formerly-aborting compute tests
  (reduction ×3, dot, norms, trace, hessenberg, strided block-width) pass on
  live hardware.

- `hephaestus-wgpu` / `hephaestus-cuda` [patch]: the blocked decomposition
  entry points (`cholesky_decompose_blocked`, `lu_decompose_blocked`,
  `qr_decompose_blocked`) now enforce a dense C-contiguous zero-offset
  operand (`validate_dense_operand`, typed `DispatchFailed`) before their
  raw whole-matrix startup copies. Previously a transposed/offset view
  computed from the wrong elements, and a broadcast (zero-stride) layout —
  whose validated storage extent is smaller than rows·cols — made the CUDA
  `cuMemcpyDtoD_v2` read past the allocation. SAFETY/inline comments at the
  copy sites now cite the check. Evidence: six new adversarial layout tests
  (transposed/offset/broadcast × both backends) pass on live hardware; full
  suites regression-free.

## [0.11.0] - 2026-07-02

ADR-0004 kernel-seam release (atlas `docs/adr/0004-hephaestus-kernel-seam.md`;
audit `docs/audit/2026-07-02-hephaestus-gpu-substrate-audit.md`). Pre-1.0
breaking minor per the versioning policy.

### Breaking

- Per-backend shader-op and scalar traits are removed: `WgslScalar`,
  `CudaScalar`, `UnaryWgslOp`/`UnaryCudaOp`, `BinaryWgslOp`/`BinaryCudaOp`,
  `ReductionWgslOp`/`ReductionCudaOp`, `ScanWgslOp`/`ScanCudaOp`,
  `ReductionIdentity`, `ScanIdentity`. One dialect-parameterized vocabulary
  in `hephaestus-core` replaces them.
- `ComputeDevice` gained required methods (`synchronize`, capability
  surface); external implementors must add them.

### Migration

- Bounds: `T: WgslScalar` → `T: DialectScalar<Wgsl>`; `T: CudaScalar` →
  `T: DialectScalar<CudaC>`; `Op: UnaryWgslOp`/`UnaryCudaOp` →
  `Op: UnaryExpr<Wgsl>`/`UnaryExpr<CudaC>` (same for Binary); reduction/scan
  ops bind `CombineExpr<L>`, identities bind `OpIdentity<Op> +
  IdentityToken<Op, L>`.
- Consts: `WGSL_TYPE`/`CUDA_TYPE` → `TYPE_TOKEN`; `WGSL_EXPR`/`CUDA_EXPR` →
  `EXPR`; `WGSL_IDENTITY`/`CUDA_IDENTITY` → `TOKEN` (literals unchanged).
- Op marker ZST import paths are unchanged (re-exported from the same
  backend modules); CUDA gains `ExpNegOp`.
- Consumer-authored kernels: implement `KernelInterface` +
  `KernelSource<L>` and dispatch via `KernelDevice::prepare`/`dispatch` or
  a `CommandStream` — see helios `GpuAttenuationMapper` for the canonical
  consumer example.

### Fixed

- `hephaestus-cuda` [patch]: kernel launches and module loads now bind the
  device context first (single `launch_kernel` SSOT) — launches from
  non-acquiring threads previously ran against the wrong or no CUDA
  context. Failed NVRTC compiles are no longer cached (transient driver
  failures no longer poison a kernel key); `SafeCachedKernel::drop` binds
  the owning context before `cuModuleUnload`; NVRTC log/PTX/destroy return
  codes are checked; stub-mode launch paths surface a typed
  `AdapterUnavailable` instead of silently succeeding.
- `hephaestus-wgpu` [patch]: `HostPinned` placement is rejected with a
  typed error on any device other than the registered staging device
  (mapped buffers belong to the creating `wgpu::Device`); uniform pool
  count raised 8→32 (2/shard starved three-uniform ops into perpetual
  reallocation); staging pool byte ceiling raised 64→512 MiB so volumetric
  readbacks pool; `prepared_axis_reduction` no longer leaks its pooled
  uniform; unary/binary storage-kernel dispatch uses pooled uniforms;
  `pinv`/`matexp` docs state their host-delegated contract.

### Performance

- `hephaestus-wgpu`/`hephaestus-cuda` [patch]: axis scan rewritten from
  O(N·L) to O(N) (one thread per scan line, combine order preserved —
  bitwise-identical results). Bench: 512x4096 f32 axis-1 cumsum
  6.07 ms → 2.29 ms (2.65x). Follow-up KS-5b files the tiled
  shared-memory variant with its derived reordering tolerance.
- `hephaestus-wgpu` [patch]: `dot`/`norm_l2`/`norm_max` fused into the
  map-reduction first pass (temp-buffer paths deleted; one less N-element
  allocation, 2N·4 B less bandwidth, one less dispatch per call).

### Changed

- `hephaestus-cuda` [patch]: repaired `--no-default-features` feature hygiene
  by making `leto-ops` a real dependency for modules that already require it
  outside `decomposition`, declaring the comparative benchmark's decomposition
  requirement, and compiling decomposition-only test helpers only with that
  feature. Evidence: `cargo check -p hephaestus-cuda --no-default-features` and
  `cargo clippy -p hephaestus-cuda --no-default-features --all-targets
  --no-deps -- -D warnings` pass.

- `hephaestus-wgpu` [patch]: completed the remaining WGPU call-site migration
  away from deleted backend-local shader traits. Linalg, random, sparse, scan
  exports, and crate exports now use shared `hephaestus_core` dialect traits
  (`DialectScalar`, `UnaryExpr`, `BinaryExpr`, `CombineExpr`, `IdentityToken`,
  `OpIdentity`) instead of `WgslScalar`/operation-specific aliases. Evidence:
  stale-name source audit is clean, `cargo check -p hephaestus-wgpu` passes,
  and `cargo clippy -p hephaestus-wgpu --all-targets --no-deps -- -D warnings`
  passes.

- `hephaestus-wgpu` [patch]: removed a stale storage-kernel `DeviceExt` import
  so downstream provider builds no longer report the unused-import warning.
  Evidence: `cargo check -p hephaestus-wgpu` passes.

### Added

- `hephaestus-core` / `hephaestus-cuda` [minor]: implemented
  `ComputeDeviceCapabilities` for the CUDA provider without fabricating
  WGPU-only limits. CUDA now snapshots driver-backed limits from
  `cuDeviceGetAttribute` / `cuMemGetInfo_v2`, reports no per-shader-stage
  storage-buffer slot limit (`None`) because CUDA authored kernels use flat
  arguments, and the no-CUDA stub remains uninhabited for capability queries.
  Evidence: `cargo fmt -p hephaestus-core -p hephaestus-wgpu -p
  hephaestus-cuda --check`, `cargo check -p hephaestus-core`, `cargo check -p
  hephaestus-wgpu`, `cargo check -p hephaestus-cuda`, `cargo check -p
  hephaestus-cuda --no-default-features`, `cargo clippy -p hephaestus-core -p
  hephaestus-wgpu -p hephaestus-cuda --all-targets --no-deps -- -D warnings`,
  `cargo clippy -p hephaestus-cuda --no-default-features --all-targets
  --no-deps -- -D warnings`, `cargo nextest run -p hephaestus-cuda
  device_capabilities_are_driver_backed` (1/1), `cargo nextest run -p
  hephaestus-cuda --no-default-features device_capabilities_are_driver_backed`
  (1/1), downstream `cargo check -p kwavers-gpu --features gpu`, `cargo
  clippy -p kwavers-gpu --features gpu --all-targets --no-deps -- -D
  warnings`, `cargo nextest run -p kwavers-gpu --features gpu backend` (31/31),
  `cargo nextest run -p kwavers-gpu --features gpu multi_gpu` (3/3), and
  `cargo nextest run -p kwavers --features gpu --test gpu_device_tests` (9/9)
  pass.

- `hephaestus-core` / `hephaestus-wgpu` [minor]: added
  `ComputeDeviceCapabilities` as the backend-neutral trait seam for enabled
  device limits and optional feature checks, plus a WGPU constructor that
  accepts `DeviceFeature` and `DeviceLimits` without exposing WGPU descriptors
  to consumers. Driver: Kwavers `WGPUContext` and `CoreGpuContext` now store a
  generic `D: ComputeDeviceCapabilities`, with WGPU only as the current default
  acquisition backend. Evidence: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu --check`, `cargo check -p hephaestus-core`, `cargo check -p
  hephaestus-wgpu`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu
  --all-targets --no-deps -- -D warnings`, downstream `cargo fmt -p
  kwavers-gpu --check`, `cargo check -p kwavers-gpu --features gpu`, `cargo
  clippy -p kwavers-gpu --features gpu --all-targets --no-deps -- -D
  warnings`, `cargo nextest run -p kwavers-gpu --features gpu backend` (31/31),
  and `cargo nextest run -p kwavers-gpu --features gpu multi_gpu` (3/3) pass.
  Follow-up on 2026-07-03 implemented the CUDA capability trait with
  driver-backed limits and `None` for WGPU-only storage-binding slots.

- `hephaestus-core` / `hephaestus-wgpu` [minor]: added backend-neutral
  `DeviceFeature` and `DeviceLimits` plus WGPU provider methods for
  feature-support checks and enabled compute limits. Also added a
  `DeviceFeature`-based acquisition constructor so consumers can request
  optional features without naming WGPU flags. Driver: Kwavers removed public
  `wgpu::Features` / `wgpu::Limits` from `GpuDevice` capability reporting.
  Evidence: `cargo fmt -p hephaestus-core -p hephaestus-wgpu --check`, `cargo
  check -p hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
  hephaestus-core -p hephaestus-wgpu --all-targets --no-deps -- -D warnings`,
  downstream `cargo check -p kwavers-gpu --features gpu`, `cargo check -p
  kwavers --features gpu --test gpu_device_tests`, `cargo clippy -p
  kwavers-gpu --features gpu --all-targets --no-deps -- -D warnings`, and
  `cargo nextest run -p kwavers --features gpu --test gpu_device_tests` pass
  (9/9).

- `hephaestus-core` / `hephaestus-wgpu` [minor]: added backend-neutral
  `DevicePreference` for GPU acquisition policy and WGPU provider constructors
  that map it to the concrete WGPU adapter preference at the backend boundary.
  Driver: Kwavers removed public `wgpu::PowerPreference` from `GpuDevice` and
  PSTD/beamforming acquisition call sites. Evidence: `cargo fmt -p
  hephaestus-core -p hephaestus-wgpu --check`, `cargo check -p
  hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
  hephaestus-core -p hephaestus-wgpu --all-targets --no-deps -- -D warnings`,
  downstream `cargo check -p kwavers-gpu --features gpu`, `cargo check -p
  kwavers-analysis --features gpu`, `cargo check -p kwavers --features gpu
  --test gpu_device_tests`, and focused downstream nextest runs pass.

- `hephaestus-wgpu` [patch]: added provider-owned `features()` and `limits()`
  accessors so downstream crates can report WGPU capability metadata without
  borrowing raw device handles. Driver: Kwavers backend contexts removed public
  raw `wgpu::Device`/`wgpu::Queue` accessors while preserving capability
  reporting. Evidence: `cargo fmt -p hephaestus-wgpu --check`, `cargo check -p
  hephaestus-wgpu`, `cargo clippy -p hephaestus-wgpu --all-targets --no-deps
  -- -D warnings`, downstream `cargo check -p kwavers-gpu --features gpu`,
  `cargo clippy -p kwavers-gpu --features gpu --all-targets --no-deps --
  -D warnings`, and `cargo nextest run -p kwavers-gpu --features gpu backend
  device multi_gpu` pass (34/34).

- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` /
  `hephaestus-metal` [minor]: added `ComputeDevice::write_sub_buffer` as the
  backend-neutral typed partial host-to-device transfer seam. WGPU and CUDA
  delegate to their concrete checked subrange transfer implementations, Metal
  delegates through its wrapped WGPU provider, and the CUDA-unavailable stub
  preserves the typed unavailable error. Evidence: `cargo fmt -p
  hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda -p hephaestus-metal
  --check`, `cargo check -p hephaestus-core -p hephaestus-wgpu -p
  hephaestus-cuda -p hephaestus-metal --all-targets --no-default-features`,
  `cargo check -p hephaestus-cuda`, `cargo clippy -p hephaestus-cuda
  --all-targets --no-deps -- -D warnings`,
  `cargo clippy -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda -p
  hephaestus-metal --all-targets --no-default-features --no-deps --
  -D warnings`, and `cargo nextest run -p hephaestus-wgpu -p hephaestus-cuda
  -p hephaestus-metal --no-default-features write_sub_buffer` pass (9/9).

- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` [minor]: added the
  backend-neutral grouped authored-kernel seam for consumers whose kernels need
  multiple WGPU bind groups while remaining flat CUDA argument lists.
  `GroupedKernelInterface` / `GroupedKernelSource<L>` declare grouped storage
  bindings, WGPU parameter group/binding, and launch shape; `GroupedKernelDevice`
  and `GroupedCommandStream` prepare and encode grouped kernels for WGPU and
  CUDA providers. WGPU builds one bind group per declared group and uses a
  provider-owned uniform parameter block; CUDA validates the same grouped
  contract and launches device-pointer arguments in declaration order plus the
  POD parameter block by value. Evidence: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu -p hephaestus-cuda --check`, `cargo check -p hephaestus-core`,
  `cargo check -p hephaestus-wgpu`, `cargo check -p hephaestus-cuda
  --no-default-features`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu
  -p hephaestus-cuda --all-targets --no-deps -- -D warnings`, `cargo nextest
  run -p hephaestus-wgpu grouped_command_stream` (2/2), and `cargo nextest run
  -p hephaestus-cuda --no-default-features cuda_grouped_command_stream` (2/2)
  pass.

- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` [minor]: added
  `GroupedKernelSequence` and `GroupedCommandStream::encode_grouped_sequence`
  for ordered grouped dispatches that must stay in one backend-defined dispatch
  region. WGPU implements the sequence as one compute pass; CUDA implements it
  as ordered launches on the bound CUDA stream. Driver: Kwavers PSTD timestep
  kernels can now migrate without splitting the existing same-pass WGPU
  dispatch order or adding a Kwavers-local helper. Evidence tier: compile-time
  validation, clippy, and value-semantic WGPU/CUDA nextest. Checks: `cargo fmt
  -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda --check`, `cargo
  check -p hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo check -p
  hephaestus-cuda --no-default-features`, `cargo check -p hephaestus-cuda`,
  `cargo clippy -p hephaestus-core -p hephaestus-cuda --all-targets
  --no-default-features --no-deps -- -D warnings`, `cargo clippy -p
  hephaestus-wgpu --all-targets --no-deps -- -D warnings`, `cargo nextest run
  -p hephaestus-wgpu stream` (8/8), and `cargo nextest run -p hephaestus-cuda
  --no-default-features stream` (6/6) pass.

- `hephaestus-wgpu` [minor]: implemented the backend-neutral
  `KernelDevice`/`CommandStream` authored-kernel seam for `WgpuDevice`. WGPU now
  prepares WGSL `KernelSource<Wgsl>` pipelines from the shared
  `KernelInterface` binding contract, records ordered dispatch/copy/zero-fill
  passes, validates typed storage bindings, and submits through the provider
  stream boundary. Evidence: `cargo fmt -p hephaestus-wgpu --check`, `cargo
  check -p hephaestus-wgpu`, `cargo clippy -p hephaestus-wgpu --all-targets
  --no-deps -- -D warnings`, and `cargo nextest run -p hephaestus-wgpu stream`
  pass (5/5). CUDA now implements the same seam through `KernelSource<CudaC>`.

- `hephaestus-core` / `hephaestus-wgpu` [minor]: added
  `MultiStorageDevice`, a backend-neutral constructor for multi-storage
  binding handles from provider-owned `D::Buffer<T>` values. `WgpuDevice`
  implements it with `WgslStorageBinding`, allowing downstream consumers to
  build multi-storage binding arrays without naming WGPU in their algorithm
  structs. Evidence: `cargo fmt -p hephaestus-core -p hephaestus-wgpu
  --check`, `cargo check -p hephaestus-core -p hephaestus-wgpu`, `cargo clippy
  -p hephaestus-core -p hephaestus-wgpu --all-targets --no-deps -- -D
  warnings`, `cargo nextest run -p hephaestus-core -p hephaestus-wgpu
  storage_kernel` (2/2), and downstream `cargo check -p kwavers-analysis
  --features gpu`, `cargo clippy -p kwavers-analysis --features gpu
  --all-targets --no-deps -- -D warnings`, `cargo nextest run -p
  kwavers-analysis --features gpu three_dimensional` (52/52) pass.

- `hephaestus-cuda` [minor]: implemented the backend-neutral
  `KernelDevice`/`CommandStream` authored-kernel seam for `CudaDevice`. CUDA now
  prepares NVRTC-compiled `KernelSource<CudaC>` kernels through the shared
  `KernelInterface` binding contract, launches typed storage bindings as
  device-pointer arguments, passes the POD parameter block by value, and
  preserves dispatch/copy/fill order through the default CUDA stream. Evidence:
  `cargo fmt -p hephaestus-cuda --check`, `cargo check -p hephaestus-cuda`,
  `cargo clippy -p hephaestus-cuda --all-targets --no-deps -- -D warnings`,
  `cargo clippy -p hephaestus-cuda --no-default-features --all-targets
  --no-deps -- -D warnings`, `cargo nextest run -p hephaestus-cuda stream`
  (3/3), and `cargo nextest run -p hephaestus-cuda --no-default-features
  stream` (3/3) pass.

- `hephaestus-core` / `hephaestus-wgpu` / `hephaestus-cuda` /
  `hephaestus-metal` [minor]: added `ComputeDevice::synchronize` as the
  backend-neutral completion seam for explicit blocking semantics. WGPU maps it
  to `Device::poll`, CUDA maps it to `cuCtxSynchronize`, Metal delegates to its
  wrapped WGPU device, and the CUDA-unavailable stub returns the existing typed
  unavailable error. Downstream Kwavers visualization transfer now uses this
  provider trait instead of raw WGPU polling.

- `hephaestus-core` / `hephaestus-wgpu` [minor]: added the backend-neutral
  `DispatchGrid`, `UnaryStorageKernel<D, T, P>`, and
  `BinaryStorageKernel<D, T, P>` kernel-dispatch contracts, plus
  `WgslUnaryStorageKernel` and `WgslBinaryStorageKernel` as WGPU
  implementations for one-input and two-input storage kernels with POD uniform
  blocks. Downstream GPU consumers can now express storage kernels over
  `ComputeDevice` implementors such as WGPU or future CUDA without owning WGPU
  pipeline construction locally.
- `hephaestus-core` / `hephaestus-wgpu` [minor]: added
  `MultiStorageKernel<D, P, B>` and the WGPU `WgslMultiStorageKernel` /
  `WgslStorageBinding` implementation for kernels with more than two storage
  buffers and one POD uniform block. This closes the WGPU provider gap for
  downstream multi-binding kernels such as Kwavers 3-D beamforming without a
  downstream bind-group helper. Evidence: `cargo check -p hephaestus-core`,
  `cargo check -p hephaestus-wgpu`, `cargo clippy -p hephaestus-core -p
  hephaestus-wgpu --all-targets -- -D warnings`, and `cargo nextest run -p
  hephaestus-core -p hephaestus-wgpu storage_kernel` pass.

- `hephaestus-cuda` / `hephaestus-python` [minor]: exposed multi-RHS sparse
  SpMV as `spmv_many`/`spmv_many_into` on CUDA and `hp.spmv_many(...)` in
  Python, both delegating to the existing sparse-dense kernel rather than a
  duplicate sparse implementation.

- `hephaestus-wgpu` [minor]: added `spmv_many`, `spmv_many_into`, and
  `prepare_spmv_many` as the public multi-RHS SpMV surface over the existing
  CSR×dense SpMM kernel. This exposes the measured GPU-preferred route for
  batched RHS vectors without duplicating sparse kernels.

- `hephaestus-wgpu` [minor]: added prepared sparse dispatch APIs,
  `prepare_spmv`/`prepare_spmm`, with `PreparedSpmv::dispatch` and
  `PreparedSpmm::dispatch` for repeated CSR products over fixed WGPU buffers.
  The prepared path reuses the pipeline, metadata uniform, and bind group across
  dispatches while preserving existing `spmv_into`/`spmm_into` one-shot APIs.
- `hephaestus-wgpu` [minor]: added `PreparedSparseDispatch` and
  `submit_prepared_sparse_batch` for one-submit batching of prepared sparse
  operations over independent output buffers.

- `hephaestus-python` [minor]: added backend-aware Python device selection via
  `Device("wgpu")` / `Device("cuda")` and routed dense linalg, sparse
  matrix-vector/matrix-matrix products, elementwise operations, reductions, and
  seeded random initializers through the selected WGPU or CUDA backend. The
  default remains WGPU for existing Python callers.

- `hephaestus-cuda` [minor]: dynamic-rank strided elementwise entry points
  (`binary_elementwise_strided_dyn_into`, `unary_elementwise_strided_dyn_into`)
  over borrowed shape/stride slices. Runtime-shaped consumers can now delegate
  rank <= 4 strided CUDA primitive binary/unary kernels through Hephaestus
  without materializing fixed-rank Leto layouts or retaining local PTX generator
  copies. Static-rank and dynamic-rank APIs share the same private launch
  helpers and cached PTX kernels.

### Tests

- `hephaestus-wgpu` [patch]: sparse CSR contract coverage now verifies
  caller-owned `spmv_into` and `spmm_into` outputs against the allocating
  `spmv`/`spmm` paths, including overwrite of pre-existing output values.

- `hephaestus-python` [patch]: extended CuPy parity coverage with a CUDA-backed
  dense-linalg path when CUDA is available and an explicit mixed-backend
  rejection test so WGPU and CUDA arrays cannot be combined silently.

- `hephaestus-cuda` [patch]: added dynamic strided CUDA value tests for
  broadcasted binary add and transposed unary sqrt, plus downstream Coeus CUDA
  live parity confirmation after routing Coeus strided primitive ops through
  the new provider surface.

- `hephaestus-wgpu` [patch]: strengthened `test_placement_aware_allocation` from
  tier-field-only checks to value semantics — Dram and Device `upload_with_hint`
  and `alloc_zeroed_with_hint` buffers are now downloaded and asserted to
  round-trip data / zero-initialize, proving a placement hint changes memory
  tier without altering values. HostPinned retains tier/length assertions: it is
  a persistently host-mapped staging buffer (`MAP_*`, no usable `COPY_SRC` while
  mapped), so it is read through its mapped pointer rather than `download`, which
  the test now documents.

### Changed

- `hephaestus-wgpu` / `-cuda` / `-metal` / `-python` [patch]: source `Complex`
  from `num_complex` directly (the layout-compatible type `leto-ops` already
  returns) after `leto` dropped its `Complex` re-export. Restores the general
  eigenvalue upload path and Python complex bindings without a cross-boundary
  type conversion; `num-complex` moved to `[dependencies]` on the crates whose
  library code references it.

- `hephaestus-wgpu` [patch]: rank-2 axis-0 reductions now use a tiled WGPU
  kernel that reduces up to 16 output columns per workgroup, replacing the
  previous one-workgroup-per-output geometry for that shape while preserving the
  generic axis API and non-axis-0 fallback.

- `hephaestus` [patch]: refreshed reduction comparative rows after the Leto
  row-major rank-2 axis-0 CPU fast path. WGPU prepared final-pass scalar sum now
  beats both Leto and `ndarray` on the local run; Leto CPU axis reductions are
  now at or near `ndarray` for the 256x256 axis-0 benchmark, while WGPU axis
  reductions remain launch/synchronization-bound at that small shape.

- `hephaestus-wgpu` [minor]: scalar reductions now use a final-pass WGSL kernel
  that lets one workgroup fold up to `BlockWidth * BlockWidth` partials,
  reducing the $2^{20}$ sum tree from three compute passes to two. Prepared
  reductions also support batch submission over independent output buffers via
  `submit_prepared_reduction_batch` and
  `submit_prepared_axis_reduction_batch`.

- `hephaestus-wgpu` [minor]: added `PreparedReduction` plus
  `prepare_reduction` and `prepare_reduction_with_width` for repeated scalar
  reductions over fixed input buffers. The prepared path reuses the compiled
  pipeline, tree scratch buffers, and bind groups; the latest comparative run
  shows this removes setup churn but does not by itself close the `ndarray`
  scalar-sum gap when repeated dispatches reuse the same scratch buffers.

- `hephaestus-wgpu` [minor]: added `PreparedAxisReduction` plus
  `prepare_sum_axis_into`, `prepare_min_axis_into`, `prepare_max_axis_into`, and
  `prepare_mean_axis_into` for repeated fixed-buffer axis reductions. The
  prepared path reuses the selected pipeline, metadata uniform, and bind group;
  comparative axis-reduction rows now measure this repeated-dispatch surface.

- `hephaestus-wgpu` [patch]: axis sum/min/max/mean now use a workgroup-memory
  tree reduction when the reduced axis fits the selected `BlockWidth`, with the
  previous scalar-per-output shader retained as the long-axis fallback. The
  comparative benchmark also accepts `HEPHAESTUS_BENCH_DISABLE_CUDA` so WGPU vs
  CPU reduction rows remain measurable on hosts where CUDA terminates the full
  harness before later sections.

- `hephaestus-wgpu` [patch]: blocked QR now downloads the first panel from the
  original input buffer before queueing the full input copy to the work buffer,
  reducing the first-panel dependency chain while preserving queue-ordered
  correctness for later in-place updates. The decomposition sync profile
  measured the QR sync floor at 213.209 µs.

- `hephaestus-wgpu` [patch]: blocked QR now reuses the Householder metadata
  uniform buffer, bind group, and host reflector-metadata scratch across panels.
  The decomposition sync profile remains transfer-bound (QR sync floor
  230.962 µs), so this removes CPU-side WGPU resource churn without claiming
  blocked-QR performance parity.

- `hephaestus-wgpu` [patch]: sparse comparative benchmarks now include the
  multi-RHS SpMV policy: repeated Leto CPU `spmv` calls are compared with the
  equivalent WGPU `spmv_many` path. On the local RTX 5080 workstation, 128 RHS
  vectors measured 62.758 µs on WGPU versus 150.414 µs for repeated Leto SpMV.

- `hephaestus-wgpu` [patch]: the focused sparse comparative benchmark now times
  the fastest measured repeated-dispatch path per sparse operation. On the
  local RTX 5080 workstation, prepared SpMV measured 61.146 µs versus Leto's
  1.232 µs, while warmed independent-output batched SpMM with the dense-RHS fast
  path measured 12.258 µs versus Leto's 35.232 µs.

- `hephaestus-wgpu` [patch]: added a C-dense RHS fast path for WGPU SpMM while
  preserving the existing generic strided kernel for non-contiguous dense views.
  The focused sparse benchmark now records the dense-RHS path at 84.978 µs
  versus Leto's 40.752 µs for the 1000x1000 CSR by 1000x128 case on the local
  RTX 5080 workstation.

- `hephaestus-wgpu` [patch]: sparse comparative benchmarks now time reusable
  caller-owned output buffers through `spmv_into`/`spmm_into` instead of
  allocating a fresh WGPU output buffer inside every timed iteration. This keeps
  the benchmark aligned with the production GPU API expected for repeated sparse
  dispatch.

- `hephaestus-cuda` [patch]: `scalar_elementwise_strided` and
  `scalar_elementwise_strided_into` now pass the scalar as a CUDA kernel argument
  instead of uploading a one-element device buffer and lowering through the
  binary strided kernel. This removes the per-call scalar storage-buffer
  allocation while preserving scalar/binary broadcast semantics.

- `hephaestus-wgpu` [patch]: strided scalar elementwise ops
  (`scalar_elementwise_strided`/`_into`) now read the broadcast scalar from a
  small pooled `uniform` via a dedicated `StridedScalarKernel`, instead of
  allocating and uploading a one-element device **storage** buffer per call and
  delegating to the binary kernel. This matches the contiguous
  `scalar_elementwise_into` pattern (SSOT for "scalar lives in a pooled
  uniform") and removes one device allocation + host→device transfer from every
  strided scalar dispatch (`hephaestus-metal` benefits too — it delegates to
  this path). Value-identical to the prior binary-broadcast lowering, verified
  by `strided_scalar_matches_binary_broadcast_semantics`. The shared strided
  metadata/decode/encode core is reused unchanged.

- `hephaestus-wgpu` [patch]: eliminated per-panel host-buffer allocations in the
  blocked Cholesky/LU/QR decompositions. The region-download SSOT is now
  `download_matrix_region_compact_into(..., out: &mut Vec<f32>)`, which reuses the
  caller's host buffer (`resize` keeps capacity) instead of returning a freshly
  allocated `Vec` each call; the old returning-`Vec` `_reusable` wrapper is
  removed (dead once all consumers migrated). Each decomposition now hoists its
  per-panel host scratch above the panel loop and refills it: LU reuses
  `col_panel`/`row_panel` (region downloads) and `diag` (sliced to the active
  `b²`), QR reuses `panel`, `packed_vectors`, and `vector_offsets`, and Cholesky
  reuses `panel`. Removes `O(n/b)` host `Vec` allocations per decomposition
  (e.g. a `b·n` row panel was ~128 KiB/panel at n=512). No behavioral change;
  device-side buffers and the host result matrix were already pre-allocated.

### Documentation

- `hephaestus-wgpu` [patch]: documented the ill-conditioned contracts of GPU
  `matrix_rank` and `det` directly on their public APIs — `matrix_rank` counts
  pivots above `relative_tolerance * max(abs(matrix))` (a pivot-magnitude
  criterion that can diverge from Leto's SVD-spectrum criterion on
  ill-conditioned inputs, agreeing when those coincide), and `det` applies no
  determinant tolerance (`relative_tolerance == 0`, so a near-singular matrix
  returns its small nonzero pivot product). Restructured `gap_audit.md` into an
  honest SSOT (Resolved / Accepted design decisions / Open future work /
  Environment) so accepted architecture and tracked future-work are no longer
  listed alongside open defects.

### Tests

- `hephaestus-wgpu` [patch]: pinned the GPU `matrix_rank` relative-threshold
  boundary (`matrix_rank_relative_tolerance_is_the_discriminator`: `diag(1,1,δ)`
  is full-rank or rank-deficient depending purely on the relative threshold and
  agrees with Leto) and the `det` near-singular contract
  (`det_of_near_singular_triangular_is_exact_pivot_product`: an upper-triangular
  input returns the exact analytical pivot product `2·3·δ`, not a
  tolerance-zeroed `0`). Closes the previously-untested ill-conditioned
  divergence residuals.

- `hephaestus-wgpu` [patch]: restored the bidiagonalization reconstruction
  contract after fixing Leto's reflector-panel factor accumulation. The focused
  WGPU contract suite now passes the documented `A = U B V^T` tall case.

### Changed

- `hephaestus-wgpu` [patch]: resolve a sub-allocated staging pointer to its
  containing mapped block in `O(log n)` instead of `O(n)`. The global
  `WGPU_MAPPED_BUFFERS` registry is now a `BTreeMap` keyed by each block's base
  address; the two HostPinned alloc/upload sites that previously held the global
  lock across an `.iter().find()` linear scan now share one `resolve_mapped_buffer`
  helper that does a `range(..=ptr).next_back()` range query plus a containment
  check, shortening the lock-held critical section under concurrent staging
  traffic. The registry and its `WgpuMappedBuffer` descriptor are tightened to
  `pub(crate)` (they have no external consumers — pure allocator-callback
  plumbing), which makes the container-type change non-breaking, and the unused
  `WgpuMappedBuffer::usage` field is removed.
- `hephaestus-wgpu` [patch]: narrowed blocked LU host/device transfers to the
  active diagonal-panel and trailing-submatrix regions, reducing full-buffer
  traffic in the hybrid GPU trailing-update path.
- `hephaestus-wgpu` [patch]: changed blocked QR to transfer compact
  trailing-column tiles per panel before GPU Householder application instead
  of writing and downloading the full working matrix.
- `hephaestus-wgpu` [patch]: packed each blocked QR panel's Householder
  vectors into one device buffer and selected the active vector by metadata
  offset, removing per-reflector vector-buffer uploads.
- `hephaestus-wgpu` [patch]: added timestamp-query profiling to the blocked
  decomposition sync benchmark to measure the blocked QR per-reflector launch
  component on the GPU timeline when the adapter supports timestamps.
- `hephaestus-wgpu` [patch]: batched blocked QR panel reflectors into one
  compute pass per panel, preserving reflector order within each column
  workgroup and removing per-reflector compute-pass launches.
- `hephaestus-wgpu` [patch]: extended the blocked decomposition sync
  benchmark with the 70x35 blocked-QR CPU panel component and removed the
  obsolete final-Leto-recompute row. The production blocked QR path constructs
  the host-side `QrDecomposition` from computed blocked factors via
  `from_raw_parts`, so profiling now targets the real synchronization floor.
- `hephaestus-wgpu` [patch]: packed blocked QR Householder vector offsets and
  beta coefficients into one reflector metadata buffer, reducing per-panel
  metadata uploads and storage bindings in the WGPU trailing-update kernel.
- `hephaestus-wgpu` [patch]: exposed an explicit transient-pool drain for
  bounded staging and uniform buffer pools so short-lived host integrations
  can release cached GPU allocations at ownership boundaries.
- `hephaestus-python` [patch]: drains WGPU transient pools when a Python
  `Device` wrapper is dropped, preventing the RNG binding test process from
  hanging after the value-semantic assertions complete.

### Added

- `hephaestus-wgpu` [minor]: device-resident CSR sparse matrix storage plus
  WGPU SpMV and SpMM kernels over packed CSR index buffers. The sparse feature
  owns the Leto CSR upload/download boundary, while the products execute on
  WGPU buffers and use the shared Mnemosyne/Moirai launch-planning path.
  Added a focused sparse comparative benchmark that validates WGPU outputs
  against Leto before reporting SpMV and SpMM timings.
- `hephaestus-wgpu` [minor]: GPU-resident linalg surface for parity with
  Leto CPU operations: allocating `matmul`/`batched_matmul`, caller-owned
  `matmul_into`/`batched_matmul_into`, dot product, trace, and L1/L2/max norms
  over strided operands. The comparative benchmark now measures WGPU against
  Leto, `ndarray`, and `nalgebra` for elementwise, reduction, matmul, dot,
  trace, and norm workloads.
- `hephaestus-wgpu` [minor]: GPU-resident Kronecker product (`kron`) over
  strided matrix operands, plus caller-owned `kron_into`, with Leto
  differential tests and comparative benchmark coverage against Leto,
  `ndarray`, and a nalgebra-backed CPU reference.
- `hephaestus-wgpu` [minor]: GPU-resident matrix power (`matpow`) over
  strided square matrix operands, using exponentiation by squaring over WGPU
  `matmul_into` dispatches, with Leto differential tests and comparative
  benchmark coverage against Leto, an `ndarray` repeated-squaring reference,
  and `nalgebra`.
- `hephaestus-wgpu` [minor]: GPU-resident finite-`f32` matrix-rank estimation
  (`matrix_rank`, `matrix_rank_with_tolerance`) using row reduction in GPU
  storage memory. Contract tests compare exact finite full-rank,
  rank-deficient, and zero matrices against Leto; comparative benchmarks cover
  WGPU, Leto, `ndarray`-backed, and `nalgebra`-backed references.
- `hephaestus-wgpu` [minor]: GPU-resident finite-`f32` determinant (`det`)
  using the shared WGPU matrix-property row-reduction dispatch. Contract tests
  compare exact finite nonsingular and singular matrices against Leto and
  reject rectangular inputs; comparative benchmarks cover WGPU, Leto,
  `ndarray`, and `nalgebra` references.
- `hephaestus-wgpu` [minor]: device-resident Cholesky, LU, and QR
  decomposition surfaces mirroring Leto's factor/solve/determinant/inverse
  APIs where applicable. Contract tests compare factors and solve/inverse
  outputs against Leto; comparative benchmarks measure WGPU API overhead
  against Leto and `nalgebra`. The current implementation delegates
  factorization to Leto on the host and stores factors on the device, so this
  is API parity rather than GPU-kernel parity.
- `hephaestus-wgpu` [minor]: device-resident SVD decomposition surfaces
  mirroring Leto's thin and rank-revealing SVD contracts. Contract tests cover
  closed-form singular values, reconstruction from downloaded `UΣVᵀ`, and
  rank-deficient rank-revealing singular values; comparative benchmarks
  measure WGPU API overhead against Leto and `nalgebra`.
- `hephaestus-wgpu` [minor]: device-resident bidiagonalization surface
  mirroring Leto's `A = U B V^T` contract. Contract tests cover orthogonal
  factors, upper-bidiagonal structure, reconstruction, singular-value
  preservation, and wide-matrix rejection; comparative benchmarks measure WGPU
  API overhead against Leto and `nalgebra` SVD.
- `hephaestus-wgpu` [minor]: device-resident Schur decomposition surface
  mirroring Leto's `A = Q T Q^T` real-Schur contract. Contract tests cover
  orthogonal factors, quasi-upper-triangular structure, reconstruction,
  spectrum preservation, and rectangular rejection; comparative benchmarks
  measure WGPU API overhead against Leto and `nalgebra` complex eigenvalues.
- `hephaestus-wgpu` [minor]: device-resident Hessenberg reduction surface
  mirroring Leto's `A = Q H Q^T` contract. Contract tests cover orthogonal
  factors, upper-Hessenberg structure, reconstruction, trace/norm similarity
  invariants, and rectangular rejection; comparative benchmarks measure WGPU
  API overhead against Leto and `nalgebra`.
- `hephaestus-wgpu` [minor]: device-resident full-pivot LU surface mirroring
  Leto's `P A Q = L U` rank-revealing contract. Contract tests cover packed
  factor reconstruction, rank reporting, determinant, solve, inverse,
  rank-deficient inverse rejection, and rectangular rejection; comparative
  benchmarks measure WGPU API overhead against Leto and `nalgebra`.
- `hephaestus-wgpu` [minor]: device-resident Bunch-Kaufman surface mirroring
  Leto's `P A P^T = L D L^T` symmetric-indefinite contract. Contract tests
  cover factor/permutation agreement, reconstruction, rectangular rejection,
  and nonsymmetric rejection; comparative benchmarks measure WGPU API overhead
  against Leto and use `nalgebra` determinant as the external CPU comparator.
- `hephaestus-wgpu` [minor]: device-resident UDU surface mirroring Leto's
  `A = U D U^T` symmetric-indefinite contract, including determinant, solve,
  and inverse methods. Contract tests cover factor agreement,
  reconstruction, solve/inverse parity, rectangular rejection, nonsymmetric
  rejection, and zero-pivot rejection; comparative benchmarks measure WGPU API
  overhead against Leto and use `nalgebra` determinant as the external CPU
  comparator.
- `hephaestus-wgpu` [minor]: baseline device-resident column-pivoted QR,
  pseudoinverse, and matrix-exponential coverage. Contract tests cover
  column-pivoted QR factor agreement plus closed-form diagonal pseudoinverse
  and matrix exponential cases; comparative benchmarks measure WGPU API
  overhead against Leto and available `nalgebra` comparators.
- `hephaestus-wgpu` [minor]: strengthened pseudoinverse and matrix exponential
  contracts with rank-deficient Moore-Penrose identities, rectangular
  pseudoinverse, non-finite rejection, nilpotent and skew-symmetric
  matrix-exponential closed forms, a general `nalgebra` exponential oracle,
  and rectangular/non-finite matrix-exponential rejection.
- `hephaestus-wgpu` [minor]: device-resident symmetric Jacobi eigen
  decomposition and eigenvalues-only surfaces mirroring Leto. Contract tests
  compare eigenvalues/eigenvectors against Leto and reject non-symmetric
  inputs; comparative benchmarks measure WGPU API overhead against Leto and
  `nalgebra`.
- `hephaestus-wgpu` [minor]: device-resident general eigenvalues for square
  `f32` matrices, returning complex device buffers. Contract coverage includes
  a diagonal closed-form eigenvalue oracle and a nonsymmetric Leto
  differential case; comparative benchmarks now measure a 32x32 block-rotation
  matrix against Leto and `nalgebra`.
- `hephaestus-wgpu` [minor]: strengthened general-eigenvalue contracts with
  exact complex-pair blocks, triangular and structured nonsymmetric spectra,
  dense `nalgebra` oracle comparison, symmetric-input real-spectrum checks,
  unordered spectrum matching, and rectangular rejection.
- `hephaestus-wgpu` [minor]: blocked Cholesky entry point
  (`cholesky_decompose_blocked`) with CPU panel factorization/solve and GPU
  SYRK trailing update. Contract coverage includes a block-boundary SPD case;
  comparative benchmarks now measure 128x128 blocked Cholesky against Leto and
  `nalgebra`.
- `hephaestus-wgpu` [minor]: comparative benchmark coverage for blocked LU and
  blocked QR GPU-trailing-update paths, with value checks against Leto before
  timing against Leto and `nalgebra`.
- `hephaestus-wgpu` [patch]: synchronization-profile benchmark for blocked
  LU/QR decomposition transfer floors, recording the current host/device
  synchronization cost before native-kernel expansion.
- `hephaestus-wgpu` [patch]: dispatch launch planning now routes through
  Mnemosyne `KernelResourceBudget` and Moirai GPU `plan_launch` while retaining
  the Hephaestus checked overflow contract.
- `hephaestus` [patch]: documented Atlas compute boundaries for Mnemosyne,
  Moirai, Themis, and Hermes. Hermes integration is through host-delegated Leto
  CPU SIMD (`leto-ops` with `simd` enabled); direct WGPU/CUDA kernel calls into
  Hermes are rejected as a boundary violation.
- `hephaestus` [patch]: consumes `moirai-gpu` with default features disabled so
  Moirai launch planning no longer pulls its optional WGPU backend into the
  Hephaestus WGPU 26 dependency graph.
- `hephaestus-wgpu` [minor]: GPU-resident rank-2 axis reductions
  (`reduce_axis`, `sum_axis`, `min_axis`, `max_axis`, `mean_axis`, and their
  caller-owned `*_into` forms) preserving Leto's rank-preserving output
  contract, with Leto differential tests and comparative benchmark coverage
  for axis 0.
- `hephaestus-wgpu` [minor]: GPU-resident rank-2 scan dispatch
  (`scan_axis_into`, `scan_axis`, `cumsum_into`, `cumsum`) with
  forward/reverse directions and cumulative sum/product markers, with Leto
  differential tests and comparative benchmark coverage for cumulative sum
  over axis 1.
- `hephaestus-wgpu` [minor]: allocating strided elementwise wrappers
  (`binary_elementwise_strided`, `unary_elementwise_strided`,
  `scalar_elementwise_strided`) returning C-contiguous GPU buffers while
  delegating to the existing caller-owned strided kernels.
- `hephaestus-wgpu` [patch]: trace and L1 norm now use fused map-reduction
  dispatches. Dot product, L2 norm, and max norm keep their staged
  map-then-reduce implementations because the fused variant regressed in the
  local comparative benchmark.
- `hephaestus-cuda` *(new crate)* [minor]: CUDA backend for the accelerator
  substrate — the GPU-side sibling of `hephaestus-wgpu`. Implements the
  `hephaestus-core::ComputeDevice` seam (`CudaDevice` acquisition + typed
  `CudaBuffer<T>` device buffer + `alloc_zeroed`/`upload`/`download` transfer),
  so consumers binding `<D: ComputeDevice>` substitute CUDA for wgpu without
  source changes. The CUDA substrate is cuda-oxide-owned for driver
  initialization, context binding, `CUdeviceptr` allocation, and transfer;
  cutile remains the kernel-authoring dependency above that substrate. The
  `cuda` feature enables the real backend; the no-default build compiles a stub
  that reports the backend unavailable rather than fabricating a device. Five
  contract tests verify
  upload/download identity (f32, i32), zeroed allocation, empty round-trip, and
  length-mismatch rejection — value-semantic, passing on real CUDA hardware
  (toolkit v13.2) and skipping when no device is present. Monomorphized kernel
  dispatch (mirroring `hephaestus-wgpu`'s `application` layer) is a follow-up.
- `hephaestus-cuda` [minor]: CUDA application-operation parity slice covering
  elementwise, strided elementwise, reductions, rank-2 axis reductions,
  rank-2 scans (`cumsum_into`/`cumsum`), matrix multiplication, Kronecker
  product, matrix power, finite-`f32` matrix rank, dot, trace, norms,
  pseudoinverse, and matrix exponential. Stub builds compile and run contract
  tests without fabricating a device; real CUDA execution remains gated behind
  the `cuda` feature and hardware availability.
- `hephaestus-cuda` [minor]: CUDA decomposition exports now include the same
  host-delegated device-resident dense wrapper families as WGPU for
  Cholesky/LU/QR, symmetric/general eigen, SVD, Schur, bidiagonalization,
  Bunch-Kaufman, full-pivot LU, Hessenberg, and UDU. Default stub-mode
  verification covers the existing contract-test slice; the newly exported
  broad dense wrappers still require operation-specific value-semantic tests
  before claiming full device parity.

### Removed

- `hephaestus-cuda` [patch]: removed stale default-build blocked-Cholesky
  export/test references; the CUDA blocked SYRK path is CUDA-feature gated and
  not claimed by the stub-mode default build.

### Fixed

- `hephaestus-wgpu` [patch]: `norm_l2` now returns the completed
  L2/Frobenius norm (`sqrt(sum(x*x))`) instead of exposing the intermediate
  squared sum, matching Leto's `norm_l2` contract.
- `hephaestus-wgpu` [patch]: uploads now allocate the same padded byte extent
  that download copies require and initialize the buffer through
  mapped-at-creation contents, preserving WGPU copy alignment without leaving a
  pending queue write on short-lived upload buffers.

## [0.10.0] - 2026-06-15

### Added

- `hephaestus-core`: `BlockWidth::checked_covering_blocks` returns `None`
  when a grid exceeds the portable `u32` block-count range.

### Changed

- `hephaestus-wgpu`: dispatch workgroup validation now uses the checked core
  grid-count API instead of duplicating overflow detection around the
  saturating helper.

## [0.9.4] - 2026-06-15

### Changed

- `hephaestus-wgpu`: scalar uniform sizing, strided metadata uniform sizing,
  and singleton reduction copy sizing now use the shared checked byte-size
  helper instead of local casts, making WGPU buffer byte sizing a single
  implementation within the crate.

## [0.9.3] - 2026-06-15

### Fixed

- `hephaestus-wgpu`: uploads now validate host slice byte size with the
  shared checked sizing helper before handing contents to WGPU buffer
  initialization, keeping allocation overflow rejection consistent across
  upload, allocation, and download paths.

## [0.9.2] - 2026-06-15

### Changed

- `hephaestus-wgpu`: binary, unary, and reduction dispatch now validate
  workgroup range before pipeline setup or intermediate buffer allocation,
  completing the dispatch precheck ordering across kernel families.

## [0.9.1] - 2026-06-15

### Changed

- `hephaestus-wgpu`: scalar and strided dispatch now validate workgroup range
  before acquiring transient uniform buffers, avoiding pool churn on impossible
  dispatch sizes.
- Added boundary coverage for the shared workgroup-count helper at the exact
  `u32::MAX` group limit and one element beyond it.

## [0.9.0] - 2026-06-15

### Changed

- `hephaestus-wgpu`: `WgpuDevice::get_staging_buffer` and
  `WgpuDevice::get_uniform_buffer` now return `Result<wgpu::Buffer>` and
  reject alignment overflow with `HephaestusError::AllocationFailed`.
- Transient staging, uniform, storage allocation, and download copy sizing now
  share checked byte-alignment arithmetic.

### Breaking

- Pre-1.0: callers of `get_staging_buffer` and `get_uniform_buffer` must handle
  `Result` instead of receiving a buffer directly.

## [0.8.1] - 2026-06-14

### Changed

- `hephaestus-wgpu`: pipeline-cache lookup now releases the cache mutex
  before first-use WGPU shader-module and compute-pipeline creation, then
  rechecks before insertion. This keeps unrelated cache access out of the
  compilation critical section.

## [0.8.0] - 2026-06-14

### Added

- `hephaestus-core`: `HephaestusError::AllocationFailed` for allocation
  requests rejected before a backend buffer is created.

### Fixed

- `hephaestus-wgpu`: storage allocation and download byte-size calculations
  now use checked arithmetic before WGPU buffer creation or copy sizing,
  rejecting impossible element counts instead of allowing wrapped byte sizes.

### Breaking

- Pre-1.0: exhaustive matches on `HephaestusError` must handle the new
  `AllocationFailed` variant.

## [0.7.3] - 2026-06-13

### Changed

- `hephaestus-wgpu`: reduction dispatch now preallocates its intermediate
  buffer handle vector from the analytically known pass count, avoiding vector
  growth while preserving the existing command-encoding lifetime model.

## [0.7.2] - 2026-06-13

### Fixed

- `hephaestus-wgpu`: `reduction_with_width` now validates power-of-two
  `BlockWidth` values before empty and singleton fast paths, so invalid
  reduction widths are rejected uniformly for every input length.

## [0.7.1] - 2026-06-13

### Added

- `hephaestus-wgpu`: `reduction_width` benchmark target comparing default
  reduction dispatch with width-128 reduction dispatch on a real adapter while
  validating both outputs against an exact host-side `u32` sum.

## [0.7.0] - 2026-06-13

### Added

- `hephaestus-wgpu`: `reduction_with_width` for caller-selected power-of-two
  `BlockWidth` reduction dispatch. The existing `reduction` API delegates to
  it with `BlockWidth::DEFAULT`.

### Changed

- Reduction WGSL generation, pipeline cache keys, intermediate output sizing,
  and dispatch group counts now use the supplied `BlockWidth` instead of a
  baked-in workgroup size.

## [0.6.9] - 2026-06-12

### Changed

- Remaining non-test invariant panic sites now use explicit
  `invariant:` messages for default block width construction and strided bind
  slot conversion.

## [0.6.8] - 2026-06-12

### Changed

- Library-code invariant panics in reduction, pipeline cache, and transient
  pool locking now use explicit `expect("invariant: ...")` messages instead
  of unqualified `unwrap()`.

## [0.6.7] - 2026-06-12

### Tests

- Remaining negative-path tests now assert concrete absence or exact mismatch
  lengths instead of broad `is_none`, `is_err`, or variant-only matches.

## [0.6.6] - 2026-06-12

### Tests

- Negative elementwise and strided dispatch tests now assert exact
  `HephaestusError` variants, lengths, and dispatch messages instead of
  accepting any error.

## [0.6.5] - 2026-06-12

### Fixed

- Caller-owned contiguous elementwise dispatch now rejects output buffers that
  alias an input before WGPU bind-group creation, producing a typed
  `DispatchFailed` contract error instead of relying on backend validation.

## [0.6.4] - 2026-06-12

### Changed

- `BoundedBufferPool::take_at_least` now selects the smallest retained buffer
  that satisfies the requested size, preserving larger retained buffers for
  later large transfers instead of consuming them for small requests.

## [0.6.3] - 2026-06-12

### Changed

- `BoundedBufferPool` now stores retained buffers in a `VecDeque`, preserving
  oldest-first count-cap eviction while avoiding `Vec::remove(0)` element
  shifts on recycle.

## [0.6.2] - 2026-06-12

### Fixed

- Transient WGPU buffer pools now evict the oldest retained buffer when the
  count cap is full, allowing the pool to adapt after larger recent staging or
  uniform allocations instead of staying polluted by smaller buffers.

## [0.6.1] - 2026-06-12

### Fixed

- Bounded transient WGPU staging and uniform buffer pools by retained count and
  bytes, preventing unbounded retained GPU memory after varied transfer and
  scalar/strided dispatch sizes.

## [0.6.0] - 2026-06-12

### Added

- Default `parallel` and `mnemosyne-memory` feature markers in
  `hephaestus-core` and `hephaestus-wgpu`, keeping the GPU substrate aligned
  with the Atlas provider feature contract without changing device dispatch.
- `hephaestus-wgpu`: `binary_elementwise_into`, `unary_elementwise_into`, and
  `scalar_elementwise_into` for caller-owned contiguous output buffers and
  non-default `BlockWidth` selection.
- `hephaestus-wgpu`: `elementwise_into` benchmark target comparing allocating
  contiguous dispatch with caller-owned output dispatch on a real adapter.

### Changed

- Contiguous elementwise allocating APIs now delegate to the caller-owned
  implementations; scalar dispatch reuses the WGPU uniform-buffer pool instead
  of allocating a uniform buffer for every call.
- Pipeline-cache construction is shared by contiguous elementwise, strided
  elementwise, and reduction kernels.

## [0.5.0] - 2026-06-11

Occupancy-planned dispatch: the strided family accepts block widths from the
ADR-0002 planner pipeline instead of a baked-in constant.

### Added

- `hephaestus-core`: `BlockWidth` — a validating `#[repr(transparent)]`
  newtype over `NonZeroU32` (zero-wide blocks are unrepresentable, not
  checked per dispatch), with `DEFAULT` (256), `covering_blocks` saturating
  ceil-division, and `Default`. This is the typed parameter through which
  moirai's occupancy planner (themis `GpuTopology` × mnemosyne
  `KernelResourceBudget`) reaches backend dispatch.
- `hephaestus-wgpu`: `StridedOperand<'_, T, N>` — a `Copy` parameter object
  pairing a device buffer with its leto layout, keeping strided signatures at
  parameter-object altitude.

### Changed (breaking, pre-1.0)

- The strided family (`binary`/`unary`/`scalar_elementwise_strided_into`)
  takes `StridedOperand` bundles plus a `BlockWidth`; WGSL is generated per
  width and the pipeline cache key is now
  `(kernel family, scalar type, width)` (`PipelineKey` alias), so widths
  cache independently and contiguous kernels share the same key space at
  their constant width.

### Tests

- On-hardware proof that a non-default width (128) produces results
  identical to the default at 1027 elements (partial trailing blocks at both
  widths), exercising per-width shader generation and cache keying. 21 tests
  total.

## [0.4.0] - 2026-06-11

ADR 0002 (atlas) provider role: device topology reporting into themis.

### Added

- `hephaestus-wgpu`: `WgpuDevice::topology()` — a themis `GpuTopology`
  snapshot captured at adapter acquisition (`try_default*` paths; the
  Arc-wrapping `new()` has no adapter and reports `None`). wgpu deliberately
  abstracts hardware topology, so only API-reported fields are filled:
  subgroup (warp/wavefront) width from adapter limits, and the memory tier
  inferred from device type (integrated → `Dram`, discrete → the
  technology-unspecified `Device` tier, since wgpu does not expose
  HBM-vs-GDDR). All other capacities are zero per themis's
  "unreported fields are zero, never fabricated" contract; the CUDA backend
  will fill the full set from device attributes.
- `themis` (0.6.0) added as a workspace dependency — hephaestus is the
  topology provider; themis stays stateless law.

### Tests

- On-hardware contract test differentially re-queries the same adapter and
  asserts warp width and tier match the API, unreported capacities are zero,
  and the adapterless constructor reports no topology.

## [0.3.1] - 2026-06-10

### Changed

- Strided dispatch meta uniforms are now pooled (`get_uniform_buffer`/
  `recycle_uniform_buffer`, mirroring the staging pool): contents are written
  with `queue.write_buffer`, which is ordered on the queue timeline relative
  to submissions, so a recycled uniform cannot race in-flight dispatches.
  Eliminates one 80-byte buffer allocation per strided dispatch.

### Docs

- ADR 0001 (Phase 2 gate): the CUDA backend composes **cuda-oxide** (device
  substrate: driver/context/streams/memory/transfers) with **cutile** (tile/
  PTX kernel authoring), preserving the no-toolkit-to-compile property, with
  a strict SoC boundary between the two and differential parity vs CPU and
  wgpu. See `docs/adr/0001-cuda-backend.md`.

## [0.3.0] - 2026-06-10

Strided op-family completion + dispatch consolidation.

### Added

- `hephaestus-wgpu`: `unary_elementwise_strided_into` — unary dispatch over
  leto layout metadata with the same broadcast/validation/caller-owned-output
  contract as the binary form.
- `hephaestus-wgpu`: `scalar_elementwise_strided_into` — **zero new kernels**:
  the scalar uploads as a one-element buffer described by an all-singleton
  leto layout, which the binary kernel broadcasts through zero strides; scalar
  semantics can never drift from binary semantics.

### Changed

- `strided.rs` consolidated to one shared core (SSOT): `StridedMeta` packing,
  WGSL `Meta`/decode fragments, `cached_pipeline`, and `encode_strided` serve
  every strided kernel family; per-family code is reduced to its shader
  expression and validation prologue.

### Tests

- Strided unary (transposed sqrt, broadcast-input neg) and scalar
  (equivalence with binary broadcast semantics over a transposed view)
  coverage; 17 tests total on real hardware.

## [0.2.0] - 2026-06-10

Phase 1 completion: strided-layout-aware dispatch over leto metadata, plus the
op-family/caching/pooling growth landed since 0.1.0.

### Added

- `hephaestus-wgpu`: `binary_elementwise_strided_into` — binary dispatch where
  all three operands are described by leto host-side `Layout<N>` (rank ≤ 4,
  compile-time capped). Inputs broadcast to the output shape with leto's own
  broadcast rules (zero-stride expanded axes; pure metadata, no data
  movement), so transposed, sliced, offset, and broadcast views dispatch with
  no contiguous materialization. The output buffer is caller-owned (allocation
  control stays with the consumer); zero-stride-aliasing output layouts are
  rejected. Shape/strides/offsets travel in one packed 80-byte uniform; the
  shader decomposes the flat index row-major and applies per-operand strides.
- `hephaestus-wgpu`: unary (`AbsOp`/`NegOp`/`ExpOp`/`LnOp`/`SinOp`/`CosOp`/
  `SqrtOp`/`RecipOp`) and scalar-broadcast elementwise dispatch; sum/min/max
  workgroup-tree reductions; pipeline caching keyed by `(kernel, T)` TypeIds;
  staging-buffer pooling on the download path.
- `leto` (core only) added as a dependency of `hephaestus-wgpu` for layout
  metadata — no CPU compute dependency (leto-ops is not pulled).

### Tests

- Strided differential suite vs CPU references over identical layout
  metadata: transposed input, dual broadcast `[2,1]+[1,3]`, offset sub-block
  write isolation, rank-3 inner-transpose, aliasing/short-buffer rejection.
  14 tests total on real hardware.

## [0.1.0] - 2026-06-10

Initial scaffold per atlas ADR 0001 (shared GPU/accelerator substrate).

### Added

- `hephaestus-core`: GPU-dependency-free contracts — `ComputeDevice` seam with
  GAT `Buffer<T: bytemuck::Pod>`, `DeviceBuffer<T>`, and a five-variant error
  vocabulary (adapter, device, length, dispatch, transfer).
- `hephaestus-wgpu`: wgpu 26 backend — `WgpuDevice` acquisition (default and
  custom limits), typed `WgpuBuffer<T>` with copy-alignment padding and a
  `raw()` escape hatch, upload/zeroed-alloc/download transfers, and
  `binary_elementwise::<Op, T>` dispatch driven by ZST `BinaryWgslOp` markers
  (`AddOp`/`SubOp`/`MulOp`) with `WgslScalar` type-token WGSL generation.
- Differential contract tests (GPU vs CPU reference) covering transfer
  round-trips, partial trailing workgroups, integral types, and length
  rejection; environment-gated skip on adapterless hosts.
