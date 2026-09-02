# ADR 0014: backend-neutral 2D Laplacian stencil parity

- Status: Accepted
- Class: [minor]
- Date: 2026-07-25

## Context

`hephaestus-wgpu` is the only accelerator backend that currently exports the
device-resident 2D Laplacian stencil. The operation's grid, boundary, and
polarity contract is backend-neutral, while the kernel launch and shader
dialect are backend-specific. CUDA and ROCm therefore have a concrete
capability gap even though both already own native multi-storage kernel seams.

## Decision

Move the POD dispatch parameters and boundary/polarity vocabulary into
`hephaestus-core`. The shared constructor validates the Leto Laplacian
contract, grid-product storage length, spacing, and f32 parameter layout.
WGPU retains its existing public paths through core re-exports. CUDA and ROCm
compile equivalent native CUDA C and HIP kernels over device-resident input
and output buffers. Metal wraps the WGPU implementation selected on its native
Metal device.

Each native kernel uses one thread per grid point and the same centered,
one-sided, and periodic finite-difference equations as the WGPU kernel. No
backend uploads storage to the host or delegates CUDA/ROCm execution through
WGPU. The common dispatch validation rejects mismatched storage before launch.

## Alternatives rejected

- Keep the stencil WGPU-only: this preserves the audited capability gap.
- Route CUDA or ROCm through WGPU: this hides vendor-native support behind an
  unrelated provider and violates backend ownership.
- Duplicate public parameter and boundary types in every backend: this lets
  validation and ABI details drift instead of giving the contract one owner.

## Verification

Core tests cover valid parameters, invalid spacing and dimensions, and the
POD parameter contract. Each backend contract compares device output with the
independent Leto CPU stencil for Dirichlet, Neumann, and periodic boundaries,
both polarity signs, and minimum/non-square grids; negative storage-length
validation is exercised through the backend entry point. Hosted CUDA, ROCm,
and Metal lanes run these contracts, while adapterless runs remain build and
typed-error evidence rather than hardware evidence.
