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

## Resolved

- [HEPH-REQUIRED-FEATURE-1] [minor] `WgpuDevice` now acquires a device only
  when it can enable every requested `DeviceFeature`; unavailable capabilities
  surface as typed acquisition failure. Apollo can require `ShaderF16` through
  Hephaestus without importing WGPU or Pollster. Evidence tier: compile-time
  feature mapping, 133-case WGPU nextest, and 196/196 applicable minor semver
  checks; real native-f16 transform evidence remains Apollo-owned.

- [HEPH-STREAM-PREFIX-1] [minor] Multilevel transform kernels no longer need a
  raw WGPU partial-copy escape hatch. `CommandStream::copy_prefix` owns the
  typed, length-checked prefix transfer in both WGPU and CUDA; the WGPU
  real-device regression verifies copied values and an unchanged suffix.

- [HEPH-TOPOLOGY-GRAPH-1] [patch] The WGPU/CUDA packages now validate against
  the default-branch Leto, Hermes, Mnemosyne, Moirai, Eunomia, and Themis
  topology graph without root patches. Evidence tier: locked
  dependency-resolution, warning-denied WGPU Clippy, value-semantic nextest,
  and warning-clean rustdoc.

- [HEPH-EMPTY-001] [patch] Deleted every synthetic singular 1x1 empty-state
  branch from the CUDA decomposition family and WGPU QR. Canonical Leto empty
  state now preserves actual shapes, identity factors, rank, permutations, and
  the empty-product determinant. The stronger optional-state redesign was
  rejected because it would duplicate state that Leto already represents.

- [major] WGPU 30 invalidated the former Mnemosyne raw-pointer staging backend:
  mapped-write memory is exposed only through explicit write-only byte views.
  Hephaestus deleted the callback registry, first-device global, mapped-pointer
  owner, and WGPU-facing Mnemosyne dependencies. Unsupported physical placement
  tiers fail at the typed boundary; provider-owned transfer pools remain the
  only WGPU staging mechanism. Evidence tier: provider type-system enforcement,
  compile-time integration, and value-semantic allocation contracts.

- [major] KS-5 reduction host planning is shared for WGPU and CUDA. Axis
  reduction metadata packing, shape/stride/output/alias validation, scalar
  reduction width validation, and scalar pass-depth planning now live in
  `hephaestus_core::reduction`; `hephaestus-wgpu` and `hephaestus-cuda` retain
  only dialect shader generation, buffer ownership, and launch mechanics.
  Evidence tier: compile-time validation, clippy, and value-semantic backend
  nextest. Checks: `cargo fmt -p hephaestus-core -p hephaestus-wgpu -p
  hephaestus-cuda --check`, `cargo check -p hephaestus-core`, `cargo check -p
  hephaestus-cuda --no-default-features`, `cargo check -p hephaestus-wgpu`,
  `cargo check -p hephaestus-cuda`, `cargo nextest run -p hephaestus-core
  reduction` (6/6), `cargo nextest run -p hephaestus-cuda
  --no-default-features reduction` (4/4), `cargo nextest run -p hephaestus-cuda
  reduction` (4/4), `cargo nextest run -p hephaestus-wgpu reduction` (5/5), and
  `cargo clippy -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda
  --all-targets --no-deps -- -D warnings`.

- [patch] CUDA Stage 1 now uses ADR-0001's cuda-oxide substrate instead of the
  previous managed-memory allocation path, and the CUDA launch SSOT drains the
  current context with a Windows-gated `cuCtxSynchronize` after every
  `cuLaunchKernel`. `CudaDevice` initializes the driver, creates and binds a
  cuda-oxide context, allocates with `cuMemAlloc_v2`, copies with checked
  `cuMemcpy*` byte counts, and frees with context-bound `cuMemFree_v2`;
  `CudaBuffer<T>` keeps `PhantomData<T>` and the owning context for typed,
  context-correct destruction. CUDA allocation hints resolve through one
  non-managed primary-buffer tier: allocatable hints record
  `MemoryTier::Device`, budget-only tiers are rejected, and
  `MappablePrimaryBuffers` is false. This resolves the former KS-8 WDDM
  `STATUS_IN_PAGE_ERROR` compute aborts. The blocked-decomposition region
  helper uses row-wise 1D copies instead of cuda-oxide 0.4.0's
  Windows-incompatible `CUDA_MEMCPY2D` layout. Evidence tier: compile-time
  validation, clippy, rustdoc, and value-semantic live-CUDA/no-driver contract
  tests. Current focused closure checks: `cargo nextest run -p hephaestus-cuda
  reduction_sum_matches_cpu_reference reduction_min_max_matches_cpu_reference
  reduction_width_is_part_of_dispatch_contract
  reduction_axis_reduction_generic_matches_cpu linalg_dot_matches_cpu_reference
  linalg_trace_matches_cpu_reference linalg_norms_match_cpu_reference
  hessenberg_reconstructs_and_preserves_similarity_invariants
  non_default_block_width_produces_identical_results` (9/9). Residual tracking
  is limited to the documented concurrent-device-acquisition case; `cargo
  nextest run -p hephaestus-cuda concurrent_device_acquisition_is_safe` passes
  1/1 on the current live-CUDA host.

- [minor] CUDA now implements the provider capability trait without fabricating
  WGPU-only descriptor values. `CudaDevice` snapshots `DeviceLimits` from real
  CUDA driver attributes and current memory info, reports `None` for
  per-shader-stage storage-buffer slots because CUDA authored kernels use flat
  arguments, and reports conservative feature support from compute capability
  and host-mapping attributes. The no-CUDA stub is uninhabited for capability
  queries, so stub mode compiles without mock limits. Evidence tier:
  compile-time validation, clippy, focused value-semantic nextest, and
  downstream integration checks. Checks: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu -p hephaestus-cuda --check`, `cargo check -p
  hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo check -p
  hephaestus-cuda`, `cargo check -p hephaestus-cuda --no-default-features`,
  `cargo clippy -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda
  --all-targets --no-deps -- -D warnings`, `cargo clippy -p hephaestus-cuda
  --no-default-features --all-targets --no-deps -- -D warnings`, `cargo
  nextest run -p hephaestus-cuda device_capabilities_are_driver_backed` (1/1),
  `cargo nextest run -p hephaestus-cuda --no-default-features
  device_capabilities_are_driver_backed` (1/1), downstream `cargo check -p
  kwavers-gpu --features gpu`, `cargo clippy -p kwavers-gpu --features gpu
  --all-targets --no-deps -- -D warnings`, `cargo nextest run -p kwavers-gpu
  --features gpu backend` (31/31), `cargo nextest run -p kwavers-gpu
  --features gpu multi_gpu` (3/3), and `cargo nextest run -p kwavers --features
  gpu --test gpu_device_tests` (9/9).

- [minor] Provider feature/limit capability checks have a trait-level
  Hephaestus seam. `ComputeDeviceCapabilities` extends `ComputeDevice` with
  enabled `DeviceLimits` and `DeviceFeature` checks, and WGPU implements the
  trait without exposing WGPU descriptors to downstream consumers. Driver:
  Kwavers `WGPUContext` and `CoreGpuContext` are generic over
  `D: ComputeDeviceCapabilities`, with WGPU only as the default acquisition
  specialization. Evidence tier: compile-time validation, clippy, downstream
  value-semantic nextest, and source audit. Checks: `cargo fmt -p
  hephaestus-core -p hephaestus-wgpu --check`, `cargo check -p
  hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
  hephaestus-core -p hephaestus-wgpu --all-targets --no-deps -- -D warnings`,
  downstream `cargo check -p kwavers-gpu --features gpu`, `cargo clippy -p
  kwavers-gpu --features gpu --all-targets --no-deps -- -D warnings`, `cargo
  nextest run -p kwavers-gpu --features gpu backend` (31/31), and `cargo
  nextest run -p kwavers-gpu --features gpu multi_gpu` (3/3). Follow-up on
  2026-07-03 implemented the CUDA capability trait with driver-backed limits
  and `None` for WGPU-only storage-binding slots.

- [minor] Device feature and limit reporting is backend-neutral at the shared
  Hephaestus boundary. `hephaestus-core::DeviceFeature` names optional
  capabilities, `DeviceLimits` names common compute limits, and
  `hephaestus-wgpu` maps them to/from WGPU only inside the provider. WGPU also
  exposes a `DeviceFeature`-based optional-feature acquisition constructor.
  Evidence tier: compile-time validation, clippy, downstream value-semantic
  nextest, and source audit. Checks: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu --check`, `cargo check -p hephaestus-core`, `cargo check -p
  hephaestus-wgpu`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu
  --all-targets --no-deps -- -D warnings`, downstream `cargo check -p
  kwavers-gpu --features gpu`, `cargo check -p kwavers --features gpu --test
  gpu_device_tests`, `cargo clippy -p kwavers-gpu --features gpu
  --all-targets --no-deps -- -D warnings`, and `cargo nextest run -p kwavers
  --features gpu --test gpu_device_tests` (9/9). Downstream source audit finds
  no `wgpu::Features`, `wgpu::Limits`, raw provider feature/limit calls, or
  WGPU-typed feature checks in `kwavers-gpu/src/gpu/device.rs` or
  `gpu_device_tests`.

- [minor] Device acquisition preference is backend-neutral at the shared
  Hephaestus boundary. `hephaestus-core::DevicePreference` now represents
  high-performance vs low-power selection policy, and `hephaestus-wgpu` maps it
  to `wgpu::PowerPreference` only inside the WGPU provider constructors.
  Evidence tier: compile-time validation, clippy, downstream value-semantic
  nextest, and source audit. Checks: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu --check`, `cargo check -p hephaestus-core`, `cargo check -p
  hephaestus-wgpu`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu
  --all-targets --no-deps -- -D warnings`, downstream `cargo check -p
  kwavers-gpu --features gpu`, `cargo check -p kwavers-analysis --features
  gpu`, `cargo check -p kwavers --features gpu --test gpu_device_tests`,
  `cargo clippy -p kwavers-gpu --features gpu --all-targets --no-deps --
  -D warnings`, `cargo clippy -p kwavers-analysis --features gpu
  --all-targets --no-deps -- -D warnings`, `cargo nextest run -p kwavers
  --features gpu --test gpu_device_tests` (9/9), `cargo nextest run -p
  kwavers-gpu --features gpu pstd_gpu` (13/13), and `cargo nextest run -p
  kwavers-analysis --features gpu three_dimensional` (52/52). Downstream source
  audit finds no `wgpu::PowerPreference` or `try_with_power_preference` use in
  Kwavers GPU, PSTD, device, or 3-D beamforming code.
- [patch] WGPU provider capability metadata is available without downstream
  raw-device borrowing. `WgpuDevice::features()` and `WgpuDevice::limits()`
  expose enabled features and limits through provider-owned methods, allowing
  consumers to remove public `wgpu::Device`/`wgpu::Queue` accessors from
  reporting-only contexts. Evidence tier: compile-time validation, clippy,
  downstream value-semantic nextest, and source audit. Checks: `cargo fmt -p
  hephaestus-wgpu --check`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
  hephaestus-wgpu --all-targets --no-deps -- -D warnings`, downstream `cargo
  check -p kwavers-gpu --features gpu`, `cargo clippy -p kwavers-gpu --features
  gpu --all-targets --no-deps -- -D warnings`, and `cargo nextest run -p
  kwavers-gpu --features gpu backend device multi_gpu` (34/34).
- [minor] Partial host-to-device buffer writes are available through the
  backend-neutral `ComputeDevice::write_sub_buffer` seam. WGPU and CUDA route
  to their checked concrete subrange transfer paths, Metal delegates through
  its wrapped WGPU provider, and the CUDA-unavailable stub preserves typed
  unavailable errors. Evidence tier: compile-time validation, clippy, and
  value-semantic backend nextest. Checks: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu -p hephaestus-cuda -p hephaestus-metal --check`, `cargo
  check -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda -p
  hephaestus-metal --all-targets --no-default-features`, `cargo check -p
  hephaestus-cuda`, `cargo clippy -p hephaestus-cuda --all-targets --no-deps
  -- -D warnings`, `cargo clippy -p
  hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda -p hephaestus-metal
  --all-targets --no-default-features --no-deps -- -D warnings`, and `cargo
  nextest run -p hephaestus-wgpu -p hephaestus-cuda -p hephaestus-metal
  --no-default-features write_sub_buffer` (9/9). Driver residual: consumers
  still need shader/command migration where they remain WGPU-specific, but
  partial transfer is no longer a concrete-backend gap.
- [minor] Grouped authored-kernel command streams and same-region sequences are
  available through the backend-neutral `GroupedKernelDevice` /
  `GroupedCommandStream` / `GroupedKernelSequence` seam. Core now declares
  grouped storage contracts with `GroupedKernelInterface` /
  `GroupedKernelSource<L>` and validates `GroupedBinding` slices by arity,
  group, binding, access, and element size. WGPU prepares grouped WGSL kernels
  with one bind group per declared group plus a uniform parameter block in the
  declared slot, and `encode_grouped_sequence` records ordered grouped
  dispatches inside one compute pass. CUDA prepares grouped CUDA C kernels,
  flattens the same storage bindings into device-pointer arguments in
  declaration order, and uses ordered launches on the bound CUDA stream for the
  same sequence contract. Evidence tier: compile-time validation, clippy, and
  value-semantic WGPU/CUDA nextest. Checks: `cargo fmt -p hephaestus-core -p
  hephaestus-wgpu -p hephaestus-cuda --check`, `cargo check -p
  hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo check -p
  hephaestus-cuda --no-default-features`, `cargo check -p hephaestus-cuda`,
  `cargo clippy -p hephaestus-core -p hephaestus-cuda --all-targets
  --no-default-features --no-deps -- -D warnings`, `cargo clippy -p
  hephaestus-wgpu --all-targets --no-deps -- -D warnings`, `cargo nextest run
  -p hephaestus-wgpu stream` (8/8), and `cargo nextest run -p hephaestus-cuda
  --no-default-features stream` (6/6). Driver residual: Kwavers PSTD still
  needs consumer shader/ABI migration and CUDA C sources, but the missing
  upstream provider seam for multi-group same-pass WGPU sequencing is closed.
- [minor] WGPU authored-kernel command streams are available through the
  backend-neutral `KernelDevice`/`CommandStream` seam. `WgpuDevice` now prepares
  WGSL `KernelSource<Wgsl>` pipelines from the shared `KernelInterface`
  contract, records ordered dispatch/copy/zero-fill passes, validates typed
  storage bindings, and submits through the provider stream boundary. Evidence
  tier: compile-time validation, clippy, and value-semantic WGPU nextest.
  Checks: `cargo fmt -p hephaestus-wgpu --check`, `cargo check -p
  hephaestus-wgpu`, `cargo clippy -p hephaestus-wgpu --all-targets --no-deps --
  -D warnings`, and `cargo nextest run -p hephaestus-wgpu stream` (5/5).
  CUDA now implements the same seam through `KernelSource<CudaC>`.
- [minor] CUDA authored-kernel command streams are available through the
  backend-neutral `KernelDevice`/`CommandStream` seam. `CudaDevice` now prepares
  NVRTC-compiled `KernelSource<CudaC>` kernels from the shared
  `KernelInterface` contract, passes typed storage bindings as device-pointer
  kernel arguments, passes the POD parameter block by value, and uses CUDA
  default-stream ordering for dispatch/copy/fill. Evidence tier: compile-time
  validation, clippy, value-semantic CUDA nextest, and no-default feature
  verification. Checks: `cargo fmt -p hephaestus-cuda --check`, `cargo check -p
  hephaestus-cuda`, `cargo check -p hephaestus-cuda --no-default-features`,
  `cargo clippy -p hephaestus-cuda --all-targets --no-deps -- -D warnings`,
  `cargo clippy -p hephaestus-cuda --no-default-features --all-targets
  --no-deps -- -D warnings`, `cargo nextest run -p hephaestus-cuda stream`
  (3/3), and `cargo nextest run -p hephaestus-cuda --no-default-features
  stream` (3/3). Residual: consumers with WGSL-only sources still need CUDA C
  kernel sources before they can run on the CUDA backend.
- [patch] WGPU stale backend-local shader trait use is removed. The remaining
  `hephaestus-wgpu` linalg, random, sparse, scan export, and crate export
  surfaces now use shared `hephaestus_core` dialect traits instead of deleted
  `WgslScalar` / per-operation alias traits, preserving the backend vocabulary
  needed for WGPU and CUDA without a compatibility shim. Evidence tier:
  compile-time validation plus static source audit. Checks: stale-name source
  audit clean, `cargo check -p hephaestus-wgpu`, and `cargo clippy -p
  hephaestus-wgpu --all-targets --no-deps -- -D warnings`.
- [minor] Backend-neutral device synchronization is available through
  `ComputeDevice::synchronize`. WGPU maps it to `Device::poll`, CUDA maps it to
  `cuCtxSynchronize`, Metal delegates to its wrapped WGPU device, and the
  CUDA-unavailable stub preserves the existing typed unavailable error. Driver
  verification: Kwavers visualization `DataPipeline<D>` now uses typed
  provider buffers plus `write_buffer`/`synchronize` without raw WGPU device,
  queue, or polling ownership. Evidence tier: compile-time validation plus
  downstream value-semantic nextest and source audit. Checks: `cargo check -p
  hephaestus-core -p hephaestus-wgpu -p hephaestus-metal`, `cargo check -p
  hephaestus-cuda`, `cargo clippy -p hephaestus-core -p hephaestus-wgpu -p
  hephaestus-cuda -p hephaestus-metal --all-targets -- -D warnings`, `cargo
  nextest run -p hephaestus-core -p hephaestus-wgpu -p hephaestus-cuda -p
  hephaestus-metal storage_kernel` (2/2), and downstream `cargo nextest run -p
  kwavers-analysis --features gpu-visualization visualization` (15/15).
- [minor] Backend-neutral multi-storage kernel dispatch is available for
  downstream consumers. `hephaestus_core::MultiStorageKernel<D, P, B>` carries
  the generic provider contract, and `hephaestus_wgpu::WgslMultiStorageKernel`
  plus `WgslStorageBinding` provide the concrete WGPU implementation for N
  storage bindings plus one POD parameter block. `MultiStorageDevice` now gives
  consumers a backend-owned storage-binding constructor from `D::Buffer<T>`,
  removing the need to name WGPU binding types in generic downstream processor
  structs. Driver verification: Kwavers 3-D static DAS and dynamic-focus DAS now
  use this provider path instead of local WGPU bind-group/compute-pass
  construction. Evidence tier: compile-time validation plus value-semantic
  layout tests and downstream beamforming nextest: `cargo check -p
  hephaestus-core`, `cargo check -p hephaestus-wgpu`, `cargo clippy -p
  hephaestus-core -p hephaestus-wgpu --all-targets -- -D warnings`, `cargo
  nextest run -p hephaestus-core -p hephaestus-wgpu storage_kernel` (2/2), and
  `cargo nextest run -p kwavers-analysis --features gpu three_dimensional`
  (52/52).
- [minor] Backend-neutral storage-kernel dispatch is available for downstream
  consumers. `hephaestus_core::DispatchGrid` covers domain extents with checked
  tile arithmetic, `UnaryStorageKernel<D, T, P>` and
  `BinaryStorageKernel<D, T, P>` dispatch over generic `ComputeDevice` buffers,
  and `hephaestus_wgpu::WgslUnaryStorageKernel` /
  `WgslBinaryStorageKernel` provide the concrete WGPU implementations for
  one-input and two-input storage kernels with uniform parameter blocks.
  Evidence tier: compile-time validation plus value-semantic grid tests: `cargo
  nextest run -p hephaestus-core kernel` (2/2), `cargo check -p
  hephaestus-core`, and `cargo check -p hephaestus-wgpu`.
- [patch] WGPU rank-2 axis-reduction occupancy is improved for short reduced
  axes. Axis sum/min/max/mean now use one workgroup per output element when the
  reduced axis length is at most the selected `BlockWidth`, reducing each output
  through workgroup memory instead of one sequential shader invocation; axis-0
  reductions now use a tiled shader that reduces up to 16 output columns per
  workgroup. The existing scalar-per-output shader remains the fallback for
  longer non-axis-0 reductions. WGPU prepared axis dispatch also reuses the
  selected pipeline, metadata uniform, and bind group for repeated fixed-layout
  reductions. Latest comparative rows: prepared final-pass scalar sum WGPU
  42.702 µs vs Leto 63.090 µs and ndarray 85.468 µs; tiled prepared axis sum
  WGPU 22.136 µs vs Leto 10.446 µs and ndarray 6.528 µs; axis min WGPU 20.726
  µs vs Leto 5.406 µs and ndarray 4.634 µs; axis max WGPU 11.778 µs vs Leto
  5.360 µs and ndarray 4.422 µs; axis mean WGPU 18.048 µs vs Leto 7.172 µs and
  ndarray 5.876 µs. Evidence
  tier: value-semantic Leto
  differential contract
  `cargo nextest run -p hephaestus-wgpu axis_reductions_match_leto_reference`
  (1/1), static diagnostics, and empirical `HEPHAESTUS_BENCH_DISABLE_CUDA=1
  cargo bench -p hephaestus-wgpu --bench comparative`.
- [patch] Blocked QR first-panel copy dependency is reduced. The first panel is
  downloaded from the original input buffer, then the full input copy to
  `work_buf` is queued so it can overlap the first CPU panel factorization
  before any write/update touches `work_buf`. Queue ordering preserves the
  subsequent in-place update semantics. Latest component profile: QR 70x35 sync
  floor 213.209 µs, CPU panel lower bound 26.369 µs, timestamp total 7.776 µs.
  Evidence tier: value-semantic blocked QR tests `cargo nextest run -p
  hephaestus-wgpu blocked_qr` (4/4), static diagnostics, and empirical `cargo
  bench -p hephaestus-wgpu --bench decomposition_sync`.
- [patch] Blocked QR Householder CPU-side WGPU resource churn is reduced. The
  Householder metadata uniform buffer, bind group, and host
  `Vec<HhReflectorMeta>` scratch are now reused across panels instead of being
  recreated in every panel with trailing columns. This preserves the existing
  one-pass panel Householder kernel and removes static CPU-side setup overhead.
  Empirical result: the synchronization profile still measures QR 70x35 at
  230.962 µs, CPU panel lower bound 28.438 µs, and timestamp total 7.904 µs, so
  the remaining bottleneck is panel transfer/synchronization rather than WGPU
  resource construction. Evidence tier: value-semantic blocked QR tests
  `cargo nextest run -p hephaestus-wgpu blocked_qr` (4/4), static diagnostics,
  and empirical `cargo bench -p hephaestus-wgpu --bench decomposition_sync`.
- [minor] Multi-RHS sparse SpMV is exposed through CUDA and Python. CUDA
  `spmv_many`/`spmv_many_into` delegate to the existing sparse-dense kernel, and
  `hephaestus-python` exposes `hp.spmv_many(...)` for WGPU and CUDA arrays.
  The in-crate Python sparse contract verifies `spmv_many` equals the SpMM
  reference output, and the SciPy parity test suite now has a `spmv_many`
  regression. Evidence tier: static diagnostics and value-semantic binding
  test `cargo nextest run -p hephaestus-python
  test_py_sparse_matrix_roundtrip_spmv_spmm` (1/1). External pytest/CuPy/SciPy
  execution remains unverified in this slice.
- [minor] WGPU multi-RHS SpMV is a public API. `spmv_many`,
  `spmv_many_into`, and `prepare_spmv_many` are thin wrappers over the existing
  sparse-dense kernel, making the GPU-preferred multi-vector route discoverable
  without duplicating WGSL or adding a parallel implementation. Latest focused
  sparse run: single prepared SpMV WGPU 61.146 µs vs Leto 1.232 µs; `spmv_many`
  with 128 RHS vectors WGPU 62.758 µs vs repeated Leto SpMV 150.414 µs; warmed
  batched prepared SpMM WGPU 12.258 µs vs Leto 35.232 µs. Evidence tier:
  value-semantic WGPU sparse contract `cargo nextest run -p hephaestus-wgpu
  --test contract test_wgpu_sparse_matrix_spmv_spmm` (1/1), `cargo check -p
  hephaestus-wgpu --bench sparse_comparative`, `cargo fmt -p hephaestus-wgpu
  --check`, and empirical `cargo bench -p hephaestus-wgpu --bench
  sparse_comparative`.
- [patch] WGPU multi-vector SpMV has an explicit performant route. The focused
  sparse benchmark now measures 128 independent RHS vectors as repeated Leto
  `spmv` calls versus the equivalent WGPU prepared SpMM dispatch, with output
  validated against Leto sparse-dense multiplication. Latest focused sparse run:
  single prepared SpMV WGPU 100.482 µs vs Leto 1.232 µs; batched SpMV via SpMM
  WGPU 34.352 µs vs repeated Leto SpMV 143.132 µs; warmed batched prepared SpMM
  WGPU 10.450 µs vs Leto 41.638 µs. Evidence tier: value-semantic benchmark
  validation, `cargo check -p hephaestus-wgpu --bench sparse_comparative`,
  `cargo fmt -p hephaestus-wgpu --check`, and empirical `cargo bench -p
  hephaestus-wgpu --bench sparse_comparative`.
- [minor] WGPU sparse batched-submit amortization is available. The closed
  `PreparedSparseDispatch` enum and `submit_prepared_sparse_batch` encode
  multiple prepared sparse products into one command buffer and submit once,
  avoiding vtable dispatch while preserving monomorphized scalar typing. The
  focused benchmark uses prepared one-shot SpMV because tiny SpMV remains
  submit-bound, and warmed independent-output batched SpMM because that regime
  amortizes submission without first-use buffer initialization. Latest focused
  sparse run: prepared SpMV 1000x1000 CSR WGPU 65.954 µs vs Leto 1.302 µs;
  warmed batched prepared SpMM 1000x1000x128 WGPU 11.940 µs vs Leto 38.466 µs.
  Evidence tier: value-semantic WGPU/Leto sparse contract `cargo nextest run -p
  hephaestus-wgpu --test contract test_wgpu_sparse_matrix_spmv_spmm` (1/1),
  `cargo check -p hephaestus-wgpu --bench sparse_comparative`, `cargo fmt -p
  hephaestus-wgpu --check`, and empirical `cargo bench -p hephaestus-wgpu
  --bench sparse_comparative`.
- [minor] WGPU sparse repeated-dispatch overhead is reduced through explicit
  prepared operations. `prepare_spmv`/`prepare_spmm` build the invariant
  pipeline, metadata uniform, and bind group once for fixed sparse/dense/output
  buffers; `PreparedSpmv::dispatch` and `PreparedSpmm::dispatch` then submit
  the real compute pass without per-call bind-group construction or uniform
  rewrites. Existing `spmv_into`/`spmm_into` remain one-shot compatibility
  paths. Latest focused sparse run: prepared SpMV 1000x1000 CSR WGPU 62.636 µs
  vs Leto 1.222 µs; prepared SpMM 1000x1000x128 WGPU 48.740 µs vs Leto
  35.498 µs. Evidence tier: value-semantic WGPU/Leto sparse contract `cargo
  nextest run -p hephaestus-wgpu --test contract
  test_wgpu_sparse_matrix_spmv_spmm` (1/1), `cargo check -p hephaestus-wgpu
  --bench sparse_comparative`, `cargo fmt -p hephaestus-wgpu --check`, and
  empirical `cargo bench -p hephaestus-wgpu --bench sparse_comparative`.
- [patch] WGPU SpMM contiguous-RHS overhead is reduced. `spmm_into` now selects a
  dedicated C-dense RHS WGSL kernel for contiguous dense operands and retains the
  existing strided kernel for non-contiguous views. The dense path removes signed
  stride arithmetic from the sparse inner loop without changing the public API or
  view semantics. Latest focused sparse run: SpMV 1000x1000 CSR WGPU 158.024 µs
  vs Leto 1.484 µs; SpMM 1000x1000x128 WGPU 84.978 µs vs Leto 40.752 µs.
  Evidence tier: value-semantic WGPU/Leto sparse contract `cargo nextest run -p
  hephaestus-wgpu --test contract test_wgpu_sparse_matrix_spmv_spmm` (1/1),
  `cargo check -p hephaestus-wgpu --bench sparse_comparative`, `cargo fmt -p
  hephaestus-wgpu --check`, and empirical `cargo bench -p hephaestus-wgpu
  --bench sparse_comparative`.
- [patch] WGPU sparse reusable-output parity is pinned. `spmv_into` and
  `spmm_into` now have value-semantic contract coverage against the allocating
  `spmv`/`spmm` paths and prove caller-owned output buffers are overwritten with
  Leto-matching values. The focused sparse benchmark now times reusable-output
  dispatch instead of allocating a new WGPU output buffer each iteration, so the
  measured GPU row reflects the intended repeated-dispatch API. Latest measured
  reusable-output rows remain below Leto parity: SpMV 1000x1000 CSR WGPU 130.940
  µs vs Leto 1.320 µs; SpMM 1000x1000x128 WGPU 88.480 µs vs Leto 36.730 µs.
  Evidence tier: value-semantic WGPU/Leto sparse contract
  `cargo nextest run -p hephaestus-wgpu --test contract
  test_wgpu_sparse_matrix_spmv_spmm` (1/1), `cargo check -p hephaestus-wgpu
  --bench sparse_comparative`, `cargo fmt -p hephaestus-wgpu --check`, and
  `cargo bench -p hephaestus-wgpu --bench sparse_comparative`.
- [patch] CUDA strided-scalar per-call storage-buffer allocation is resolved.
  `scalar_elementwise_strided`/`_into` now pass the scalar as a CUDA kernel
  argument through a dedicated scalar strided kernel instead of uploading a
  one-element device buffer and delegating through binary strided dispatch. The
  shared strided metadata/decode path and scalar/binary broadcast semantics are
  preserved. Evidence tier: static diagnostics and value-semantic CUDA strided
  contracts on available CUDA runtime (`cargo check -p hephaestus-cuda`;
  `cargo fmt -p hephaestus-cuda --check`; `cargo nextest run -p hephaestus-cuda
  --test strided` -> 11/11).
- [patch] Bidiagonalization factor-contract parity is restored. The failing
  tall case came from local Leto factor accumulation reading reflector panels
  with the wrong layout convention when returning `U`/`V`; Hephaestus only
  exposed the provider output after upload. Leto now applies the reflector-major
  panel slices sequentially for returned factors, preserving the documented
  `A = U B V^T` contract. Evidence tier: value-semantic provider contract
  `cargo nextest run -p leto-ops bidiagonalize_tall` (1/1) and WGPU contract
  suite `cargo nextest run -p hephaestus-wgpu --test contract` (94/94).
- [minor] WGPU `matrix_rank` ill-conditioned threshold contract is now pinned and
  documented. The kernel counts pivots greater than
  `relative_tolerance * max(abs(matrix))`; this is a row-reduction (pivot
  magnitude) criterion rather than Leto's SVD-spectrum criterion, so the two can
  diverge on ill-conditioned inputs where pivot magnitudes and singular values
  differ. The boundary behaviour — same near-singular matrix is full-rank or
  rank-deficient depending purely on the relative threshold, and agrees with Leto
  when pivot magnitudes equal singular values (diagonal/orthogonally-scaled
  inputs) — is documented on `matrix_rank_with_tolerance` and pinned by
  `matrix_rank_relative_tolerance_is_the_discriminator`. Evidence tier:
  value-semantic threshold-boundary contract test plus Leto differential.
- [minor] WGPU `det` ill-conditioned contract is now pinned and documented. `det`
  passes `relative_tolerance == 0`, so only an exactly-zero pivot collapses the
  determinant to zero; a near-singular matrix returns its small, nonzero pivot
  product (no determinant-tolerance zeroing), which can diverge from an
  SVD/tolerance-based determinant on ill-conditioned inputs. Documented on `det`
  and pinned by `det_of_near_singular_triangular_is_exact_pivot_product` (exact
  analytical pivot product `2·3·δ` on an upper-triangular input, plus Leto
  differential). Evidence tier: analytically-derived value-semantic test.
- [patch] Hephaestus WGPU launch planning uses Mnemosyne `KernelResourceBudget`
  and Moirai GPU `plan_launch` through Moirai's planner-only feature set. The
  prior duplicate-WGPU risk is closed: Hephaestus depends on `moirai-gpu` with
  default features disabled, so `moirai-gpu` no longer pulls `wgpu 0.19` into the
  Hephaestus graph. Evidence tier: dependency-tree verification and package
  checks.
- [patch] Python binding tests previously hung after the RNG binding test body
  completed because transient WGPU staging/uniform buffers could remain retained
  across short-lived host-runtime teardown. `PyDevice` now drains the bounded
  WGPU transient pools on drop. Evidence tier: root-cause diagnostic run showing
  the test body completed before process hang, Python package nextest, and full
  workspace nextest.
- [patch] WGPU staging-allocator registry (`WGPU_MAPPED_BUFFERS`) contention
  audit (2026-06-23). The HostPinned alloc/upload paths resolve a Mnemosyne
  sub-allocated pointer to its containing wgpu mapped block. This previously held
  the global registry lock across an `O(n)` `.iter().find()` linear scan over
  every live mapping (`O(n²)` over a staging-heavy workload). Replaced the
  `HashMap` with a base-address-keyed `BTreeMap` and consolidated both sites into
  one `resolve_mapped_buffer` helper doing an `O(log n)`
  `range(..=ptr).next_back()` containment query, shortening the lock-held section.
  Remaining bounded characteristic (not a defect): a single global `Mutex` still
  serializes the registry across threads; the lookup is now `O(log n)` and the
  `wgpu::Buffer` return is an `Arc` handle clone, so the critical section is
  minimal. The deeper lever, if a future staging-heavy multi-thread workload
  measures the global lock as hot, is a sharded registry keyed by address range —
  deferred until a workload demonstrates the need (no current consumer drives
  concurrent HostPinned staging). The registry and its descriptor are now
  `pub(crate)` (no external consumers) and the dead `usage` field is removed.
  Evidence tier: value-semantic placement/transfer contract tests + full
  workspace gate.

- [patch] Blocked-decomposition per-panel host-allocation churn (audit
  2026-06-23). The blocked Cholesky/LU/QR panel loops allocated a fresh host
  `Vec` every panel for each region download (`col_panel`/`row_panel`/`panel`,
  up to `b·n` ≈ 128 KiB/panel at n=512) plus per-panel scratch (LU `diag`, QR
  reflector-packing buffers) — `O(n/b)` host allocations per call on top of the
  already-pooled device buffers. Resolved: the region-download SSOT is now
  `download_matrix_region_compact_into(out: &mut Vec)` (reuses host capacity via
  `resize`); the dead returning-`Vec` `_reusable` wrapper is removed; each
  decomposition hoists its per-panel host scratch above the loop and refills it.
  The remaining blocked-decomposition perf gap is the host/device
  synchronization floor and native-kernel parity (tracked under Open future
  work), not host allocation. Evidence tier: cross-block-boundary value-semantic
  contract tests (blocked LU n=66, blocked Cholesky/QR across boundary) + full
  workspace gate.

- [patch] Strided scalar per-call storage-buffer allocation (audit 2026-06-23).
  `scalar_elementwise_strided`/`_into` uploaded the broadcast scalar to a fresh
  one-element device **storage** buffer every call (`device.upload(from_ref)`)
  and delegated to the binary kernel — a per-call device allocation + transfer,
  inconsistent with the contiguous `scalar_elementwise_into` path which already
  reads the scalar from a pooled `uniform`. Resolved with a dedicated
  `StridedScalarKernel` reading the scalar from a pooled uniform (no per-call
  storage operand), reusing the shared strided metadata/decode/encode core;
  value-identity verified by `strided_scalar_matches_binary_broadcast_semantics`.
- [info] Storage-buffer pooling examined and deferred (audit 2026-06-23). Unlike
  the uniform/staging pools, `alloc_zeroed` always creates a fresh device buffer,
  so multi-pass reductions allocate `O(log n)` intermediate buffers per call.
  These are **not** trivially poolable: all passes of a reduction are encoded in
  one command buffer and submitted once (no per-pass sync), so every intermediate
  must stay alive until the single submit completes — they cannot be freed/reused
  within a call. Cross-call recycling would require fence-based deferred return
  (the readback path is fire-and-forget, no `poll(Wait)`), so a naive pool would
  recycle buffers the GPU is still reading (UAF). A safe storage pool is a real
  but non-trivial change with uncertain payoff (the blocked-decomposition
  profile shows the host/device sync floor, not allocation, dominates); deferred
  until a workload measures storage-allocation churn as material.

## Accepted Design Decisions

- WGPU/Leto parity is complete for the current core array-operation slice:
  elementwise, strided elementwise, scalar elementwise, reductions, rank-2 axis
  reductions, rank-2 scans, matrix products, Kronecker product, matrix power,
  finite-`f32` matrix rank, finite-`f32` determinant, dot, trace, norms,
  Cholesky/LU/full-pivot LU/QR/column-pivoted-QR/SVD/bidiagonalization/Schur/Hessenberg/Bunch-Kaufman/UDU
  decomposition APIs, pseudoinverse and matrix exponential baseline wrappers,
  symmetric Jacobi eigen decomposition/eigenvalue APIs, and general eigenvalues
  for diagonal closed-form and nonsymmetric Leto-differential cases. Evidence
  tier: value-semantic contract tests against CPU references and Leto, plus
  comparative benchmark evidence in `benchmark_results.md`. (Per-operator
  GPU-kernel/performance parity for the host-delegated wrappers is tracked under
  Open future work.)
- [minor] Hermes integration is intentionally host-tier for Hephaestus:
  host-delegated parity wrappers call `leto-ops` with `simd` enabled, and Leto
  routes CPU hot loops through Hermes SIMD before Hephaestus uploads verified
  outputs into device buffers. Direct WGPU/CUDA kernel calls into Hermes are out
  of scope because Hermes owns CPU SIMD over host slices while Hephaestus owns GPU
  resource lifetimes and device-resident kernels. Evidence tier:
  dependency/implementation audit and ADR 0002.

## Open Future Work — GPU-kernel & performance parity

These surfaces are API-complete and value-verified against Leto/nalgebra; the
remaining gaps are native-GPU-kernel and/or performance parity (`[major]`
effort), not correctness. Factorization/solve currently delegate to Leto on the
host before uploading device buffers.

- [arch] CUDA multi-storage beamforming dispatch is still a future concrete
  provider implementation. The generic trait and WGPU implementation are
  delivered; adding CUDA requires a real CUDA beamforming kernel and launch path,
  then downstream Kwavers verification against that provider. No Kwavers helper
  layer is required.
- [patch] WGPU axis reductions still carry fixed dispatch/synchronization
  overhead against CPU backends on small workloads after the short-axis
  workgroup reduction path, axis-0 tiling, prepared dispatch, batched axis
  submission, scalar final-pass collapse, and Leto's row-major rank-2 axis-0 CPU
  fast path. Current residual: scalar sum beats `ndarray` on the latest run, but
  Leto CPU axis reductions remain faster than WGPU for 256x256 axis 0. Definition
  of ready for the next reduction slice: prototype a measured small-axis routing
  policy or fuse multiple axis statistics into one WGPU pass; do not target
  Hermes SIMD arithmetic until a CPU-profile shows the arithmetic loop rather
  than layout/launch overhead is dominant. Evidence tier: value-semantic
  contract plus empirical comparative benchmark.
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
  10.0 µs. Remaining risk: blocked QR still trails Leto/`nalgebra`; the next
  measured lever is reducing the host/device synchronization count, not metadata
  buffer count or CPU panel arithmetic. Evidence tier: value-semantic blocked QR
  tests, empirical synchronization/component profiles, comparative benchmark, and
  GPU-timeline timestamp measurement in `benchmark_results.md`.
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
- [patch] `cuda-oxide` 0.4.0's build script links `cuda.lib`, so the default
  `hephaestus-cuda` feature needs `CUDA_LIB_PATH` pointing at a CUDA import
  library even though adapterless runtime tests skip and the
  `--no-default-features` stub compiles without a CUDA driver/device. This is an
  upstream build-script constraint, not a Hephaestus runtime-device contract.

## Next Increment

- Continue the parity audit at the next highest-risk Open future-work residual:
  reduce scalar-reduction pass/submit overhead via Hephaestus/Mnemosyne launch
  planning or add a measured small-reduction CPU-routing policy before returning
  to single-vector SpMV and blocked QR synchronization.
