# hephaestus-core

GPU-dependency-free contracts for the Atlas shared accelerator substrate (atlas
ADR 0001). This crate defines *what a compute device is* — device acquisition
results, typed device buffers, and the dispatch seam — without depending on any
GPU API, so code generic over a backend compiles on a machine with no
accelerator.

Backend crates (`hephaestus-wgpu`, `hephaestus-cuda`, `hephaestus-rocm`) and the
CPU reference (`hephaestus-host`) implement these contracts. Consumers (`apollo`
GPU transforms, `coeus` GPU tensor backends) program against the seam so
spectral and tensor packages share one device layer without an `apollo`→`coeus`
dependency edge. Autodiff stays in `coeus`; kernels dispatched here are
autodiff-agnostic functions.

Most consumers should depend on the `hephaestus` facade, which re-exports this
crate flat.

## What it defines

- `ComputeDevice` — the extension seam, deliberately not sealed, with a GAT
  `Buffer<T: Pod>` so backends substitute without consumer changes. Consumers
  bind generically (`<D: ComputeDevice>`) and dispatch is monomorphized; no
  `dyn` on hot paths.
- `DeviceBuffer<T>` — typed buffer contract. Element types are bounded by
  `eunomia::Pod` and dtype lives in `PhantomData<T>`, so dtype confusion is a
  compile error.
- Shared backend-neutral parameter vocabulary: volume ray geometry and
  validation, 2D Laplacian parameters and boundary conditions.
- Scalar-generic `FftOps` with validated dense split-complex operands, fixed
  forward/inverse normalization, one const-rank prepared contract for 1D, 2D,
  and 3D, and a provider-neutral command-stream encoding boundary for composed
  kernels.
- Runtime-rank `DynamicStridedView`, `FusedExpression`, and separate
  elementwise/reduction fusion seams. Providers own expression lowering and
  validate borrowed Leto layouts without copying storage.
- The error vocabulary, including distinct allocation rejection.

`#![forbid(unsafe_code)]`.

## Documentation

- API reference: [docs.rs/hephaestus-core](https://docs.rs/hephaestus-core)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
