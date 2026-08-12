# CUDA and ROCm Backends

Hephaestus provides CUDA and ROCm backends through `hephaestus-cuda` and
`hephaestus-rocm` respectively. Both implement the full op trait surface
and expose device-native APIs where available.

## CUDA Backend (`hephaestus-cuda`)

CUDA kernels are compiled at build time using `cuda-build`. The kernel
language is `CudaC` (C++ with CUDA extensions):

```rust,ignore
use hephaestus_cuda::CudaDevice;

let device = CudaDevice::acquire(0)?;  // acquire CUDA device 0
```

The CUDA backend supports:
- Device memory allocation through `MemoryTier::Gddr`
- Host-pinned memory for DMA staging (`MemoryTier::HostPinned`)
- CUDA unified memory (`MemoryTier::Device`)
- cuBLAS for dense matrix operations where available

## ROCm Backend (`hephaestus-rocm`)

The ROCm backend uses HIP kernels (`KernelDialect::HipC`):

```rust,ignore
use hephaestus_rocm::RocmDevice;

let device = RocmDevice::acquire(0)?;
```

ROCm supports AMD GPU devices (RDNA, CDNA architectures). The backend
provides rocBLAS for dense matrix operations where available.

## Metal Backend (`hephaestus-metal`)

`hephaestus-metal` targets Apple Silicon and discrete AMD GPUs on macOS.
Metal kernels use MSL (Metal Shading Language).

## Expression Parity

GELU, LGAMMA, and error function expressions have been verified to produce
matching results across all four backends (WGPU, CUDA, ROCm, Metal) with
hosted CI evidence from Coeus PRs #228 and #231.
