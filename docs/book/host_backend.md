# Host Backend

`hephaestus-host` is the pure-CPU backend for Hephaestus. It implements
all op traits using Mnemosyne-allocated host RAM, enabling development and
testing without a GPU.

## Purpose

The host backend is useful for:
- Unit testing kernels without GPU hardware
- CPU-only deployments where GPU resources are unavailable
- Fallback execution when device acquisition fails

## Activation

```rust,ignore
use hephaestus_host::HostDevice;

let device = HostDevice::new();
```

`HostDevice` implements the full `ElementwiseOps`, `DenseProductOps`,
`AxisReductionOps`, and other op traits. All computation runs on the CPU
using Mnemosyne for buffer allocation.

## `DeviceBuffer<T>` on Host

Host buffers are heap-allocated `Vec<T>`-backed slices managed by
Mnemosyne. `StridedView` is resolved to contiguous copies for op dispatch
on the host backend.

## Decomposition Backend

`hephaestus-host` also provides the blocked-panel decomposition backend
for CPU-side factorizations (LU, Cholesky, QR, SVD) via
`BlockedDecompositionBackend`.
