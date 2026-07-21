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
| `hephaestus-wgpu` | Portable wgpu backend (wgpu 26): adapter/device acquisition, typed `WgpuBuffer<T>` (PhantomData-typed over `wgpu::Buffer`), upload/download with pooled staging, and monomorphized elementwise/reduction dispatch via ZST op markers + per-`(Op, T, BlockWidth)` WGSL generation. |
| `hephaestus-cuda` | CUDA backend: cuda-oxide device acquisition, context binding, `CUdeviceptr` allocation, typed `CudaBuffer<T>`, host/device transfer, and monomorphized elementwise/reduction/scan/linalg/sparse dispatch via ZST op markers and cutile kernel authoring. Dynamic-rank strided elementwise entry points let runtime-shaped consumers delegate their GPU tensor layout kernels without depending on Coeus-local CUDA generators. |
| `hephaestus-python` | Thin PyO3/NumPy boundary over the Rust WGPU and CUDA device APIs. |

## Python Releases

GitHub Releases tagged `hephaestus-python-v<version>` build locked CPython
3.9–3.13 wheels for Linux, Windows, and macOS. The workflow installs and
imports each wheel as `pyhephaestus`, verifies that its `hephaestus-python`
metadata version matches the release tag, attests and attaches the exact wheel
set to the GitHub Release, then publishes those same artifacts to PyPI through
OIDC Trusted Publishing. The tag version must equal the workspace Cargo
version.

## Design

- The `ComputeDevice` trait is the deliberate extension seam — not sealed;
  backends substitute without consumer changes. Consumers bind generically
  (`<D: ComputeDevice>`); dispatch is monomorphized, no `dyn` on hot paths.
- Element types are bounded by `bytemuck::Pod`; buffer dtype lives in
  `PhantomData<T>` so dtype confusion is a compile error.
- Elementwise kernels follow leto-ops' ZST operation-marker pattern on the
  device side: generic allocating APIs delegate to caller-owned `*_into`
  entry points, and the op contributes only its shader combine expression. No
  type names appear in API identifiers (`WgslScalar::WGSL_TYPE` /
  `CudaScalar::CUDA_TYPE` substitutes the shader type token).
- Contiguous and strided elementwise callers can supply output buffers, so
  allocation policy stays with the consumer. Contiguous outputs must not alias
  inputs; scalar dispatch reuses the same uniform-buffer pool as strided
  metadata.
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
and does not inherit Moirai's optional WGPU backend.

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
cargo bench --bench elementwise_into
cargo bench --bench reduction_width
```

Contract tests run real device dispatch differentially against CPU references
(upload/download round-trip, partial trailing workgroup add, integral mul,
length-mismatch rejection). On hosts without an adapter the tests skip with a
message rather than fabricate a pass.

The `elementwise_into` and `reduction_width` benchmarks run real WGPU dispatch
and validate output values. They are empirical timing tools, not Criterion
regression baselines.

## Consumers

- `apollo`: `apollo-wgpu-helpers` delegates adapter/device acquisition here
  (public API preserved for the 16+ `-wgpu` transform crates).
- `coeus`: GPU `ComputeBackend` implementations re-base here when coeus bumps
  to wgpu 26 (currently on 23); tracked in coeus MS-60+ Stage D.
