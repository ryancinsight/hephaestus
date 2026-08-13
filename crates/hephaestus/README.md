# hephaestus

The entry crate for the Atlas accelerator substrate. Depend on this one; the
sub-crates exist so a consumer *can* take a narrower dependency, not because it
should.

`hephaestus` is a facade and owns no logic. It re-exports the device, buffer,
transfer, and kernel contracts from `hephaestus-core` at facade paths
(`hephaestus::DeviceBuffer`), and each backend behind a feature under a module
named for its device API (`hephaestus::wgpu`, `hephaestus::cuda`,
`hephaestus::rocm`).

Conceptually, hephaestus is to the GPU what `leto` is to the CPU: a shared
buffer and compute substrate that lets high-level packages (`apollo` spectral
transforms, `coeus` tensor backends) share device contexts and allocations
without depending on each other.

## Contracts compile anywhere

The contract layer is always present and pulls in no device stack, so code
generic over a backend builds on a machine with no accelerator at all:

```rust
use hephaestus::DeviceBuffer;

fn total_elements<T, B: DeviceBuffer<T>>(buffers: &[B]) -> usize {
    buffers.iter().map(|buffer| buffer.len()).sum()
}
```

## Install

```toml
# contracts only
hephaestus = "0.19"

# portable compute
hephaestus = { version = "0.19", features = ["wgpu", "decomposition", "sparse"] }

# NVIDIA; needs a CUDA toolkit at build time for headers
hephaestus = { version = "0.19", features = ["cuda", "decomposition"] }
```

## Feature flags select what is compiled

No backend is enabled by default: a default backend would make every consumer of
the contracts pull a device stack, and `cuda`/`rocm` additionally require vendor
toolkits at build time.

Enabling a backend feature compiles it in. The facade performs no backend
selection — it contains no logic. Select a backend explicitly by naming its
device type (`WgpuDevice::try_default`, `CudaDevice::try_default`, …); each
returns a typed unavailable-device error on a host without that hardware, so a
caller wanting a preference order writes that order itself.

| Feature | Effect |
| --- | --- |
| `wgpu` | portable compute backend; the usual starting point |
| `cuda` | NVIDIA backend; CUDA toolkit needed at build time |
| `rocm` | AMD backend; HIP system libraries needed at build time |
| `metal` | acquires a Metal-preferring wgpu device (see ADR 0047) |
| `decomposition` | dense Cholesky/LU/QR on the enabled backends |
| `sparse` | CSR upload/download and GPU sparse products |
| `parallel`, `mnemosyne-memory` | forwarded to the contract layer and every enabled backend |

## Documentation

- API reference: [docs.rs/hephaestus](https://docs.rs/hephaestus)
- Workspace overview, layer boundaries, and verification policy: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
