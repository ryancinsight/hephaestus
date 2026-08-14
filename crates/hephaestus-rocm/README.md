# hephaestus-rocm

Native AMD ROCm/HIP device substrate for the Atlas accelerator stack. It
implements the shared `hephaestus_core::ComputeDevice` seam over the Linux HIP
runtime, so consumers that bind generically (`<D: ComputeDevice>`) substitute
ROCm for wgpu or CUDA without source changes. Most consumers reach it through
the `hephaestus` facade as `hephaestus::rocm`.

## What it provides

- Linux HIP device acquisition, driver-backed limits and topology, typed
  `RocmBuffer<T>`, transfer, and synchronization.
- hipRTC-compiled and module-launched kernels across the operation families:
  contiguous and rank-≤4 strided elementwise, reductions and rank-2 axis
  reductions, scans, map-reductions, Kronecker products, matrix powers, matrix
  properties, tiled and batched matrix multiplication, and CSR sparse products.
  Sparse storage owns values, column indices, and row pointers in typed device
  buffers, and multi-RHS SpMV reuses the SpMM kernel rather than duplicating it.
- Native HIP scaled dot-product attention, mean cross-entropy, and regular and
  transposed convolution with additive gradients.
- `RocmMultiStorageKernel` and `RocmCommandStream` over HIP's ordered default
  stream, implementing the shared multi-storage and grouped-kernel seams.

## Requirements and features

The `rocm` feature enables the Linux HIP runtime implementation and needs ROCm
system libraries. Without that feature the crate remains buildable on hosts
without ROCm and `RocmDevice::try_default` returns a typed unavailable-device
error — it does not fall back to WGPU or CPU.

`decomposition` adds the device-resident factorization surface: Cholesky,
partial/complete-pivot LU, Householder and column-pivoted QR, bidiagonalization,
SVD, UDU, Bunch–Kaufman, Hessenberg, real Schur, and symmetric and general
eigen decompositions.

## Documentation

- API reference: [docs.rs/hephaestus-rocm](https://docs.rs/hephaestus-rocm)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
