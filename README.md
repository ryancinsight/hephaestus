# Hephaestus: Atlas GPU/Accelerator Substrate

Hephaestus is the shared GPU device substrate for the Atlas stack (atlas ADR
0001, `atlas/docs/adr/0001-gpu-accelerator-substrate.md`). It owns device/
context/queue acquisition, typed device buffers, and a `ComputeDevice`
dispatch seam with per-backend implementations, so spectral (`apollo`) and
tensor (`coeus`) packages share one device layer without an `apollo`→`coeus`
dependency edge.

## Naming

Hephaestus is the god of the forge: the place where the stack's compute
kernels are forged for accelerator hardware.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `hephaestus-core` | GPU-dependency-free contracts: `ComputeDevice` seam (GAT `Buffer<T: Pod>`), `DeviceBuffer<T>`, error vocabulary. `#![forbid(unsafe_code)]`. |
| `hephaestus-wgpu` | Portable wgpu backend (wgpu 26): adapter/device acquisition, typed `WgpuBuffer<T>` (PhantomData-typed over `wgpu::Buffer`), upload/download with pooled staging, and monomorphized elementwise/reduction dispatch via ZST op markers + per-`(Op, T, BlockWidth)` WGSL generation. |

Planned sibling backend: CUDA, **composing `cuda-oxide`** (driver/runtime/
device-memory/streams) **with `cutile`** (tile/PTX kernel authoring),
preserving the dynamic-load / no-toolkit-to-compile property.

## Design

- The `ComputeDevice` trait is the deliberate extension seam — not sealed;
  backends substitute without consumer changes. Consumers bind generically
  (`<D: ComputeDevice>`); dispatch is monomorphized, no `dyn` on hot paths.
- Element types are bounded by `bytemuck::Pod`; buffer dtype lives in
  `PhantomData<T>` so dtype confusion is a compile error.
- Elementwise kernels follow leto-ops' ZST operation-marker pattern on the
  device side: generic allocating APIs delegate to caller-owned `*_into`
  entry points, and the op contributes only its WGSL combine expression. No
  type names appear in API identifiers (`WgslScalar::WGSL_TYPE` substitutes
  the shader type token).
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
allocation (mnemosyne — planned device pools/pinned staging), ownership
proofs (melinoe — planned device-buffer tokens), or scheduling (moirai).

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run
cargo test --doc
cargo doc --no-deps
cargo bench --bench elementwise_into
```

Contract tests run real device dispatch differentially against CPU references
(upload/download round-trip, partial trailing workgroup add, integral mul,
length-mismatch rejection). On hosts without an adapter the tests skip with a
message rather than fabricate a pass.

The `elementwise_into` benchmark runs real WGPU dispatch and validates output
values. It is an empirical timing tool, not a Criterion regression baseline.

## Consumers

- `apollo`: `apollo-wgpu-helpers` delegates adapter/device acquisition here
  (public API preserved for the 16+ `-wgpu` transform crates).
- `coeus`: GPU `ComputeBackend` implementations re-base here when coeus bumps
  to wgpu 26 (currently on 23); tracked in coeus MS-60+ Stage D.
