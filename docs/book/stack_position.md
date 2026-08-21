# Position in the Stack

## Hephaestus owns

Hephaestus owns the accelerator boundary: `ComputeDevice`,
`ComputeDeviceCapabilities`, `ComputeDeviceAcquisition`,
`DeviceBuffer<T>`, the typed operation seams, prepared dispatch contracts,
backend device implementations, and the shared conformance clauses.

The operation families documented here include `ElementwiseOps`,
`FullReductionOps`, `AxisReductionOps`, `DenseVectorOps`, and
`DecompositionOps`. The operation traits are generic over a device and
scalar where the contract supports that variation; the backend implementation
is selected at the operation boundary and is monomorphized by Rust.

Hephaestus does not own tensor semantics, uncertainty laws, transform-domain
algorithms, scheduling policy, or memory-placement vocabulary. Those concerns
remain in their owning Atlas crates and cross this boundary through typed
contracts such as `themis::PlacementHint` and `themis::GpuTopology`.

## Dependency direction

```text
themis placement vocabulary
          |
          v
hephaestus-core: device, buffer, and operation contracts
          |
          +--> hephaestus-host   (reference implementation)
          +--> hephaestus-wgpu   (portable GPU implementation)
          +--> hephaestus-cuda   (CUDA implementation)
          +--> hephaestus-rocm   (ROCm implementation)
```

Consumers depend on the core contracts or a selected backend crate. They do
not duplicate device acquisition, buffer ownership, or operation validation.
`hephaestus-conformance` runs the same value-semantic clauses against each
available implementation. Metal selection is part of `hephaestus-wgpu`, not a
separate implementation in this graph. A provider-specific test is not a
substitute for the shared oracle.
