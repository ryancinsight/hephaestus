# hephaestus-cuda

CUDA backend for the Atlas shared accelerator substrate (atlas ADR 0001). It is
the GPU-side sibling of `hephaestus-wgpu`: it implements the same
`hephaestus_core::ComputeDevice` seam, so consumers that bind generically
(`<D: ComputeDevice>`) substitute CUDA for wgpu without source changes. Most
consumers reach it through the `hephaestus` facade as `hephaestus::cuda`.

## What it provides

- Device acquisition, context binding, `CUdeviceptr` allocation, typed
  `CudaBuffer<T>`, and host/device transfer through cuda-oxide.
- Kernel authoring above that substrate through cutile.
- Monomorphized elementwise, reduction, scan, map-reduction, linalg, sparse,
  volume, and Laplacian dispatch through the shared ZST operation markers, with
  native compiled kernels retained for prepared plans and repeated dispatch.
- Dynamic-rank strided elementwise entry points, so runtime-shaped consumers can
  delegate their GPU tensor layout kernels rather than carrying their own CUDA
  generators.
- Device-resident decomposition results, including an exact packed-LU split
  and lazy QR Q accumulation from compact Householder factors. R-only and
  least-squares QR consumers do not allocate the `m × m` orthogonal factor.

## Requirements and features

The `cuda` feature enables the native backend and needs a CUDA toolkit at build
time for headers. Without it the crate still compiles, and
`CudaDevice::try_default` reports the backend unavailable rather than
fabricating a device.

`decomposition` implies `cuda`: it is not an independent host-only surface, so
Cargo enables the device substrate its kernels launch on.

## Documentation

- API reference: [docs.rs/hephaestus-cuda](https://docs.rs/hephaestus-cuda)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
