# Changelog

SemVer 2.0.0; pre-1.0 minor bumps may include breaking changes (documented).

## Unreleased

### Changed

- `hephaestus-wgpu` [patch]: narrowed blocked LU host/device transfers to the
  active diagonal-panel and trailing-submatrix regions, reducing full-buffer
  traffic in the hybrid GPU trailing-update path.
- `hephaestus-wgpu` [patch]: changed blocked QR to transfer compact
  trailing-column tiles per panel before GPU Householder application instead
  of writing and downloading the full working matrix.

### Added

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
  source changes. The CUDA toolchain is composed from cutile-rs
  (`cuda-core` driver `sys` + `cuda-async` device acquisition), the same source
  coeus-cuda uses, gated behind the `cuda` feature and dynamically loaded at
  runtime; the default build compiles a stub that reports the backend
  unavailable rather than fabricating a device. Five contract tests verify
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
