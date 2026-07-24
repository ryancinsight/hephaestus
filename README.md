# Hephaestus: Atlas GPU/Accelerator Substrate

Hephaestus is the shared GPU device substrate for the Atlas stack (atlas ADR
0001, `atlas/docs/adr/0001-gpu-accelerator-substrate.md`). It owns device/
context/queue acquisition, typed device buffers, and a `ComputeDevice`
dispatch seam with per-backend implementations, so spectral (`apollo`) and
tensor (`coeus`) packages share one device layer without an `apollo`→`coeus`
dependency edge.

Conceptually, **Hephaestus is to the GPU what [`leto`](../leto) is to the CPU**. Much like `leto` serves as the non-differentiable CPU array vocabulary, `hephaestus` serves as the shared GPU buffer and compute substrate. It decouples high-level packages (such as `apollo` spectral transforms and `coeus` tensor backends) so they can share device contexts and memory allocations without direct dependencies, mirroring the role of **CuPy** in the Python (NumPy/SciPy) ecosystem.

## Naming

Hephaestus is the god of the forge: the place where the stack's compute
kernels are forged for accelerator hardware.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `hephaestus-core` | GPU-dependency-free contracts: `ComputeDevice` seam (GAT `Buffer<T: Pod>`), `DeviceBuffer<T>`, and distinct error vocabulary including allocation rejection. `#![forbid(unsafe_code)]`. |
| `hephaestus-wgpu` | Portable wgpu backend (wgpu 30): adapter/device acquisition, typed `WgpuBuffer<T>` (PhantomData-typed over `wgpu::Buffer`), upload/download with pooled staging, and monomorphized elementwise/reduction dispatch via ZST op markers + per-`(Op, T, BlockWidth)` WGSL generation. |
| `hephaestus-cuda` | CUDA backend: cuda-oxide device acquisition, context binding, `CUdeviceptr` allocation, typed `CudaBuffer<T>`, host/device transfer, and monomorphized elementwise/reduction/scan/linalg/sparse dispatch via ZST op markers and cutile kernel authoring. Dynamic-rank strided elementwise entry points let runtime-shaped consumers delegate their GPU tensor layout kernels without depending on Coeus-local CUDA generators. |
| `hephaestus-rocm` | Native AMD ROCm/HIP backend: Linux HIP device acquisition, driver-backed limits/topology, typed `RocmBuffer<T>`, transfer/synchronization, and hipRTC/module-launched contiguous and rank-≤4 strided binary, unary, and scalar elementwise operations, contiguous sum/min/max, rank-2 axis sum/min/max/mean reductions, rank-2 prefix/suffix scans, tiled rank-2 and batched matrix multiplication, strided Kronecker products, matrix powers, finite rank estimation, determinants, seeded uniform/normal initializers, strided dot/trace/L1/L2/max norms, device-resident CSR matrices, HIP SpMV/SpMM dispatch, backend-neutral multi-storage HIP kernels, authored-kernel streams with grouped sequencing, and optional HIP Cholesky, partial/complete-pivot LU, Householder/column-pivoted QR, Golub–Kahan bidiagonalization, and SVD decomposition surfaces. Enable the optional `rocm` feature on a ROCm host; add `decomposition` for the factorization surface. |
| `hephaestus-python` | Thin PyO3/NumPy boundary over the Rust WGPU and CUDA device APIs. |

## Python Releases

GitHub Releases tagged `hephaestus-python-v<version>` build locked CPython
3.9–3.13 wheels for Linux, Windows, and macOS. The workflow installs and
imports each wheel as `pyhephaestus`, verifies that its `hephaestus-python`
metadata version matches the release tag, attests and attaches the exact wheel
set to the GitHub Release, then publishes those same artifacts to PyPI through
OIDC Trusted Publishing. The tag version must equal the workspace Cargo
version. Published wheels enable the portable WGPU backend. CUDA entry points
remain present and return the typed backend-unavailable error; source builds on
CUDA 13.2+ hosts opt into the native backend with the `cuda` feature.

## Rust Crate Releases

The `Crates.io Release` workflow validates a named workspace package on manual
dispatch. After that package's required first release is published locally and
its crates.io Trusted Publisher is registered, a GitHub Release tagged
`crate-<package>-v<version>` packages, verifies, and publishes the matching
Cargo version with a short-lived OIDC token. Validation runs in a separate
read-only job. The publish job is bound to the GitHub `crates-io` environment;
register each package's Trusted Publisher with that environment.
`hephaestus-python` remains a wheel-only artifact and is marked
`publish = false` for crates.io.

## Design

- The `ComputeDevice` trait is the deliberate extension seam — not sealed;
  backends substitute without consumer changes. Consumers bind generically
  (`<D: ComputeDevice>`); dispatch is monomorphized, no `dyn` on hot paths.
- Element types are bounded by `bytemuck::Pod`; buffer dtype lives in
  `PhantomData<T>` so dtype confusion is a compile error.
- Elementwise kernels follow leto-ops' ZST operation-marker pattern on the
  device side: generic allocating APIs delegate to caller-owned `*_into`
  entry points, and the op contributes only its shader combine expression. No
  type names appear in API identifiers; `DialectScalar<L>::TYPE_TOKEN`
  substitutes the shader type token for each backend dialect.
- The ROCm backend owns native HIP device mechanics through the optional
  `cubecl-hip-sys` bindings. Its `HipC` dialect reuses the shared operation
  vocabulary, while real contiguous and rank-≤4 strided elementwise, reduction, rank-2 axis-scan,
  map-reduction, and matrix multiplication sources compile through hipRTC and
  launch through the HIP module API. Contiguous reduction partials stay in
  typed device buffers across host-planned tree passes; rank-2 axis reductions
  and scans consume shared core shape/stride plans directly; matrix
  multiplication uses a 16×16 shared-memory tile and the same strided
  layout/alias validation as CUDA and WGPU. Batched matrix multiplication uses
  the grid-z dimension and singleton-batch broadcasting. Dot, trace, and norm
  kernels use one rank-four packed layout contract for transposed, sliced, and
  diagonal views. Kronecker products use one HIP thread per logical output
  coordinate over the same strided matrix metadata. Matrix powers use
  exponentiation by squaring over device-resident matrices, copying strided
  inputs through the native identity elementwise kernel before reusing the
  tiled matmul path. Matrix rank and determinant use one HIP workgroup for
  Gaussian elimination over a packed strided view, with the same tolerance
  contract as CUDA and WGPU. Seeded random initializers explicitly delegate
  deterministic value generation to `leto-ops`, matching CUDA and WGPU before
  uploading the result to ROCm storage. The strided elementwise
  family decodes one packed rank-four logical index for binary, unary, and
  scalar operations, including Leto broadcast views. Its default build has no
  ROCm linkage and returns a typed
  unavailable-device error instead of falling back to WGPU or CPU.
- Sparse ROCm storage owns CSR values, column indices, and row pointers in
  typed device buffers. HIP SpMV and SpMM kernels consume that representation
  directly in `O(nnz)` and `O(nnz * rhs_columns)` work; multi-RHS SpMV reuses
  the SpMM kernel rather than duplicating the sparse implementation.
- The ROCm decomposition feature runs finite-input validation, Cholesky
  diagonal/column recurrences, partial/complete-pivot LU steps, and
  Householder/column-pivoted QR steps through ordered HIP kernels. Bidiagonalization
  and SVD mirror the CUDA/WGPU shared Leto provider boundary and upload typed
  factors and singular values to ROCm buffers. The ordinary entry points
  accept strided rank-2 inputs through the native strided identity kernel; the
  blocked entry points preserve the dense C-contiguous contract shared with
  CUDA and WGPU. Solve, determinant, inverse, and least-squares methods retain
  the existing host-side scalar contract after HIP factorization; no backend
  selection fallback to CPU or WGPU is used.
- `RocmMultiStorageKernel` implements the shared `MultiStorageKernel` and
  `MultiStorageDevice` contracts with flat HIP pointer arguments plus a POD
  parameter block. Binding order, arity, block dimensions, and length
  mismatches are rejected before a real HIP module launch.
- `RocmCommandStream` implements `KernelDevice` and `CommandStream` over HIP's
  ordered default stream. `RocmGroupedPrepared` and grouped sequences preserve
  the shared grouped-kernel ABI while retaining flat HIP pointer arguments;
  device copies and byte fills use HIP driver operations rather than host
  materialization.
- Contiguous and strided elementwise callers can supply output buffers, so
  allocation policy stays with the consumer. Contiguous outputs must not alias
  inputs; scalar dispatch reuses the same uniform-buffer pool as strided
  metadata.
- `prepare_dot` and `prepare_norm_l2` bind fixed input buffers once and retain
  their scalar output and reduction-tree scratch buffers. Repeated dispatches
  observe writes to those buffers without reallocating or rebuilding bind
  groups. L2 norm encodes its map, reduction tree, and square root into one
  command buffer and submits once.
- Staging and uniform buffer pools are bounded by retained count and retained
  bytes, keeping transient GPU memory reuse from becoming unbounded growth.
- `WgpuBuffer::raw()` is the consumer escape hatch: apollo transform kernels
  build their own pipelines/bind groups over hephaestus-allocated storage.

## Layer boundary

Hephaestus owns device acquisition, device buffers, transfer, and generic
dispatch. It does **not** own: autodiff (coeus), transform kernels (apollo),
CPU arrays (leto — whose host-side `Layout<N>` metadata it reuses), host
allocation/resource-budget vocabulary (mnemosyne), ownership proofs (melinoe —
planned device-buffer tokens), thread-level scheduling (moirai), or CPU SIMD
(hermes). WGPU launch sizing uses Mnemosyne `KernelResourceBudget` and Moirai
GPU `plan_launch` through Moirai's planner-only feature set; acquired devices
expose Themis topology snapshots. Hephaestus owns its concrete WGPU 26 runtime
and does not inherit Moirai's optional WGPU backend. Native HIP device
mechanics, elementwise kernels, reductions, scans, map-reductions, Kronecker
products, matrix powers, matrix properties, seeded random initializers, tiled
matrix multiplication, CSR sparse products, multi-storage kernels, and
authored-kernel streams belong to `hephaestus-rocm`.

Hermes integration is intentionally indirect for host-delegated Leto parity
wrappers: Hephaestus depends on `leto-ops` with its `simd` feature enabled, and
Leto routes CPU hot loops through Hermes SIMD before Hephaestus uploads the
results to device buffers. Native WGPU/CUDA kernels do not call Hermes because
Hermes is the Atlas CPU SIMD substrate over host slices, while Hephaestus owns
GPU resource lifetimes and device-resident shader/PTX kernels. See
[`docs/adr/0002-atlas-compute-boundaries.md`](docs/adr/0002-atlas-compute-boundaries.md).

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run
cargo test --doc
cargo doc --no-deps
cargo check -p hephaestus-rocm --features rocm --all-targets --locked
cargo clippy -p hephaestus-rocm --features rocm --all-targets --no-deps -- -D warnings
cargo nextest run -p hephaestus-rocm --features rocm --locked
cargo test -p hephaestus-rocm --features rocm --doc --locked
cargo bench --bench elementwise_into
cargo bench --bench reduction_width
cargo run -p hephaestus-wgpu --example prepared_map_reduction
cargo bench -p hephaestus-wgpu --bench prepared_map_reduction
```

Contract tests run real device dispatch differentially against CPU references
(upload/download round-trip, partial trailing workgroup add, integral mul,
length-mismatch rejection). On hosts without an adapter the tests skip with a
message rather than fabricate a pass.

The ROCm contract suite runs HIP allocation, zeroing, upload/download,
subrange-write, length-rejection, capability, topology, binary/unary/scalar
elementwise, rank-≤4 strided binary/unary/scalar elementwise, contiguous
sum/min/max, rank-2 axis sum/min/max/mean reduction,
rank-2 forward/reverse scan, tiled/batched matrix-multiplication, strided
Kronecker products, matrix-power, matrix-rank/determinant, seeded uniform and
normal initializers, CSR round-trip, SpMV, and SpMM value checks, and strided
dot/trace/norm value checks, multi-storage HIP binary dispatch, and authored
kernel stream/grouped-sequence copy/fill/value checks. With
`rocm,decomposition`, it also runs Cholesky, pivoted-LU, pivoted-QR,
bidiagonalization, SVD factor, strided-input, blocked-density, solve, determinant,
inverse, empty-input, and failure contracts. The ROCm
container CI lane
validates the feature build and adapterless path; the
manually enabled self-hosted AMD lane sets
`HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` so hardware evidence cannot be replaced by
a skip.

The `elementwise_into`, `reduction_width`, and `prepared_map_reduction`
benchmarks run real WGPU dispatch and validate output values. The prepared
map-reduction benchmark compares fixed-buffer prepared dot/L2 dispatch against
the same one-shot operations at identical inputs and Criterion settings. These
are empirical timing tools, not stored regression baselines.

## Consumers

- `apollo`: `apollo-wgpu-helpers` delegates adapter/device acquisition here
  (public API preserved for the 16+ `-wgpu` transform crates).
- `coeus`: GPU `ComputeBackend` implementations re-base here when coeus bumps
  to wgpu 26 (currently on 23); tracked in coeus MS-60+ Stage D.
