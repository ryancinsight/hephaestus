# Position in the Stack

## What Hephaestus Owns

Hephaestus is the Atlas accelerator seam. It owns:

- **Device acquisition** — `ComputeDevice`, `ComputeDeviceAcquisition`
- **Typed command streams** — `CommandStream<K>` for kernel dispatch
- **Op trait contracts** — `ElementwiseOps`, `DenseProductOps`,
  `AxisReductionOps`, `ConvolutionOps`, `AttentionOps`,
  `DecompositionOps`, and 15 other traits
- **Plan types** — `AttentionPlan`, `ConvolutionPlan`, etc.
- **Device buffer abstraction** — `DeviceBuffer<T>`
- **All backend implementations** — wgpu, CUDA, ROCm, Metal, host

Hephaestus does **not** own tensor semantics (Coeus), transform algorithms
(Apollo), runtime scheduling (Moirai), or memory allocation policy (Mnemosyne).

## Where Hephaestus Sits

`	ext
mnemosyne (allocator) + themis (placement vocab)
  |
  v
hephaestus-core (op trait contracts)
  |                   |
  v                   v
hephaestus-wgpu   hephaestus-cuda   hephaestus-rocm   hephaestus-metal
  |
  v (consumed by)
coeus-wgpu   apollo-fft(wgpu)   moirai-gpu
`

## Consumers

| Consumer | How Hephaestus is used |
|----------|------------------------|
| `coeus-cuda/wgpu/rocm/metal` | ML tensor op dispatch |
| `apollo-fft` | GPU transform backend (`WgpuTransformBackend`) |
| `moirai-gpu` | GPU task scheduling and stream management |

## Themis Feedback Loop

Hephaestus reads device properties at acquisition time and creates
`GpuDeviceProperties` for Themis `GpuTopology`. Moirai then reads
`GpuTopology` to decide whether to route tensor operations to GPU workers.

## `hephaestus-conformance`

The conformance crate runs the complete op-trait test suite against every
backend to verify parity. All backends must pass the same oracle tests
before a release.
