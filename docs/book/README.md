# hephaestus — GPU Accelerator Substrate for Atlas

`hephaestus` is the GPU accelerator substrate of the Atlas stack.  It defines
the `ComputeDevice` contract, typed `DeviceBuffer`, and operation seams
(`ElementwiseOps`, `DecompositionOps`, `DenseProductOps`) that accelerator
backends implement.  Backends — `wgpu` (portable), `cuda` (NVIDIA), `rocm`
(AMD) — are feature-gated; no backend is enabled by default.

## Design goals

- **Backend-neutral contracts** — `hephaestus-core` has no GPU API dependency;
  code generic over `ComputeDevice` compiles without any vendor toolkit.
- **HostDevice for testing** — `hephaestus-host` implements `ComputeDevice`
  over plain host memory, so every conformance test and every `book_*.rs`
  example runs on a machine without a GPU.
- **Type-level correctness** — buffer lengths are checked at every boundary;
  the only way to produce a `DeviceBuffer` is via the device's constructor, so
  buffer ownership is tied to the device that allocated it.

## What this book covers

1. The `ComputeDevice` contract and buffer lifecycle.
2. `DeviceBuffer`: allocation, upload, download, sub-buffer write.
3. Device capabilities and limits.
4. Operation seams: elementwise, reduction, decomposition.
5. The `HostDevice` reference backend.
6. The wgpu, CUDA, and ROCm backends.
7. Where hephaestus fits in the Atlas compute stack.
