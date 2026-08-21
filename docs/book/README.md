# hephaestus — GPU Accelerator Substrate for Atlas

`hephaestus` is the accelerator substrate of the Atlas stack. It defines the
`ComputeDevice` contract, typed `DeviceBuffer<T>`, and operation seams such
as `ElementwiseOps`, `FullReductionOps`, `AxisReductionOps`, and
`DecompositionOps` that backend crates implement. The workspace keeps the
portable WGPU, CUDA, ROCm, and host implementations separate from the core
contract crate; Metal selection is a WGPU adapter preference.

## Design goals

- **Backend-neutral contracts** — `hephaestus-core` has no GPU API dependency;
  code generic over `ComputeDevice` compiles without any vendor toolkit.
- **HostDevice for testing** — `hephaestus-host` implements `ComputeDevice`
  over plain host memory, so transfer and generic conformance tests run on a
  machine without a GPU.
- **Type-level correctness** — the associated `ComputeDevice::Buffer<T>` type
  ties a buffer to its backend and element type; transfer methods validate
  logical lengths before copying.

## What this book covers

1. The `ComputeDevice` contract and buffer lifecycle.
2. `DeviceBuffer`: allocation, upload, download, sub-buffer write.
3. Device capabilities and limits.
4. Operation seams: elementwise, reduction, decomposition.
5. The `HostDevice` reference backend.
6. The wgpu, CUDA, and ROCm backends.
7. Where hephaestus fits in the Atlas compute stack.
