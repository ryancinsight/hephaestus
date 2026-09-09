# hephaestus-cuda

CUDA backend for the Atlas shared accelerator substrate (atlas ADR 0001). It is
the GPU-side sibling of `hephaestus-wgpu`: it implements the same
`hephaestus_core::ComputeDevice` seam, so consumers that bind generically
(`<D: ComputeDevice>`) substitute CUDA for wgpu without source changes. Most
consumers reach it through the `hephaestus` facade as `hephaestus::cuda`.

## What it provides

- Device acquisition, context binding, `CUdeviceptr` allocation, typed
  `CudaBuffer<T>`, and host/device transfer through its owned CUDA driver ABI.
- Runtime CUDA kernel compilation through NVRTC above that substrate.
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

The `cuda` feature enables the native backend without build-time CUDA headers
or import libraries. Execution needs an NVIDIA driver; runtime kernel
compilation additionally needs NVRTC. Without the feature the crate compiles, and
`CudaDevice::try_default` reports the backend unavailable rather than
fabricating a device.

Dense matrix, batched matrix and Kronecker products support `f32`, `f64`,
`i32`, `u32`, `eunomia::F16` and `eunomia::Bf16`. Arithmetic and accumulation
stay in the selected scalar. Half arithmetic requires compute capability
5.3 or later; Bf16 requires 8.0 or later. Compilation targets the acquired
device and rejects lower capabilities instead of widening the arithmetic.
Header-dependent scalar compilation requires toolkit headers beside the loaded
NVRTC library (`bin`, `bin/x64`, `lib64`, or `lib/<target>` installations).
Missing headers produce compilation errors; header-free kernels need only NVRTC.

Scalar extensions declare source headers through
`hephaestus_core::DialectScalar<CudaC>::PRELUDE`. Implementations and qualified
calls using the former `CudaFusionScalar::PRELUDE` move to that trait; see
[ADR 0044](../../docs/adr/0044-device-neutral-dense-product-seam.md).

`decomposition` implies `cuda`: it is not an independent host-only surface, so
Cargo enables the device substrate its kernels launch on.

## Documentation

- API reference: [docs.rs/hephaestus-cuda](https://docs.rs/hephaestus-cuda)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
