# ADR 0012 (hephaestus): native ROCm backend through HIP

- Status: Accepted
- Date: 2026-07-24
- Class: [arch] — introduces a new accelerator backend crate
- Parent decision: atlas ADR 0001 (`atlas/docs/adr/0001-gpu-accelerator-substrate.md`)

## Context

Hephaestus already exposes one backend-neutral `ComputeDevice` seam with WGPU,
CUDA, and Metal implementations. AMD hardware needs a native ROCm path for
workloads that require HIP runtime/device semantics rather than WGPU's portable
Vulkan path. The provider must remain buildable on hosts without ROCm and must
not report an AMD device when HIP cannot acquire one.

The first complete vertical slice was the device substrate: HIP device
acquisition, typed device memory, host/device transfer, synchronization,
capability limits, and Themis topology. The bounded operator slices now have a
consumer acceptance oracle: contiguous and rank-≤4 strided binary, unary, and
scalar elementwise operations, contiguous and rank-2 axis reductions, rank-2 scans, rank-2/3
matrix multiplication, strided Kronecker products, matrix powers, matrix
rank/determinant, and strided map-reductions
can be compared against CPU values and the existing CUDA/WGPU contracts.

## Decision

Add `hephaestus-rocm` as a sibling backend crate. Its `rocm` feature enables
the published `cubecl-hip-sys` raw bindings on Linux; the default feature set
contains no ROCm dependency and exposes the same typed unavailable-device
behavior as the existing CUDA stub. `RocmDevice` maps HIP runtime calls to the
existing `ComputeDevice`, `ComputeDeviceCapabilities`, and
`ComputeDeviceAcquisition` traits. `RocmBuffer<T>` owns a HIP device allocation,
stores the logical element count and placement tier, and carries `T` through
`PhantomData`.

HIP's current-device selection is thread-local, so every allocation, transfer,
synchronization, module load, kernel launch, and drop binds the buffer's
recorded ordinal before calling HIP. The backend uses ordinary `hipMalloc`
device memory; host-visible or managed placement hints normalize to the
implemented `Device` tier, while budget-only tiers are rejected. Device limits
and topology are queried from HIP attributes and memory information;
unsupported acquisition is surfaced as a typed error. Elementwise sources use
the shared `HipC` dialect and compile through hipRTC, then load one cached HIP
module entry point per `(operation, scalar, block width)` key. The strided
family shares one packed rank-four metadata/decode contract for binary, unary,
and scalar kernels, maps Leto broadcast views by zeroing singleton-axis
strides, and validates output buffers as distinct from inputs with no
zero-stride races, matching the CUDA/WGPU elementwise contract.
Contiguous reductions use the same cached module shape for each tree pass and
retain typed partial buffers until the final one-element result is returned;
empty inputs return the typed operation identity. Rank-2 axis reductions use
the shared `AxisReductionMeta` and `plan_axis_reduction` contracts, so shape,
stride, storage, alias, and empty-axis validation remains provider-neutral.
Rank-2 scans use the shared `AxisScanMeta` and `plan_axis_scan` contracts. One
HIP block owns each logical scan line, lanes fold contiguous chunks in order,
and shared-memory chunk prefixes complete forward or reverse cumulative sums
and products without a second provider-specific host planner.
Rank-2 matrix multiplication uses one 16×16 HIP block per output tile and
shared-memory tiles over the contracted dimension. `matmul_into` accepts the
same strided rank-2 layouts as CUDA and WGPU, validates storage and output
aliasing before launch, and zero-fills partial edge tiles so non-multiple
matrix dimensions have the same value contract.
The batched form dispatches the batch dimension through grid-z, treats a
singleton input batch as a zero batch stride, and chunks launches at the HIP
grid-z limit so batches are not serialized into one launch per matrix.
Dot products, traces, and L1/L2/max norms share one rank-four packed metadata
contract. Each workgroup maps its logical strided view into shared memory and
the existing contiguous reduction kernel finishes multi-workgroup inputs;
L2 applies the HIP square-root elementwise kernel to the reduced scalar.
Kronecker products use one HIP thread per logical output coordinate. The kernel
decomposes that coordinate into left/right matrix coordinates and applies the
three strided rank-2 layouts, so output tiling and non-contiguous views do not
require a provider-specific host copy.
Matrix powers use exponentiation by squaring: a strided input is copied into
contiguous device storage through the native identity elementwise kernel, the
identity result handles exponent zero, and every product reuses the tiled
matmul implementation. No CPU or WGPU fallback participates in the operation.
Matrix rank and determinant use one single-thread HIP workgroup for sequential
Gaussian elimination over a device scratch buffer. The kernel copies logical
strided coordinates into row-major scratch storage, applies partial pivoting,
uses the relative tolerance only for rank, and returns determinant zero for
non-square or rank-deficient inputs. The public validation and scalar contract
match CUDA and WGPU.
Seeded uniform and normal initializers follow the existing CUDA/WGPU
host-delegated contract: `leto-ops` owns deterministic random-value generation,
and ROCm owns the typed device upload. This boundary is explicit in the API and
does not silently substitute CPU execution for a requested HIP kernel.
The module cache is thread-confined with the HIP current-device binding because
HIP module handles and device pointers are not cross-thread Rust values.

## Alternatives rejected

- WGPU-only AMD support: it provides portable Vulkan execution but does not
  expose the native ROCm/HIP runtime contract requested here.
- A consumer-owned AMD wrapper: it duplicates the provider seam and makes
  HIP ownership downstream of Hephaestus.
- A CPU or WGPU fallback behind the ROCm feature: it would hide missing HIP
  capability and violate the provider's typed-unavailable contract.
- Implementing the full CUDA/WGPU operator surface in one ROCm increment: it
  would couple unrelated kernel families and obscure which parity contracts
  have real AMD hardware evidence. Elementwise and contiguous reduction are
  bounded families; each later family gets its own CPU-differential contract.

## Consequences and verification

The new crate is Linux/ROCm-native and does not promise Windows or macOS HIP
support. CI always checks the default, ROCm-featured, and adapterless paths in
a ROCm development container. A manually enabled self-hosted AMD runner runs
the same contract suite with `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1`, so a skipped
hardware test cannot be mistaken for device evidence. The current ROCm parity
surface is contiguous elementwise, contiguous sum/min/max reduction, rank-2
axis sum/min/max/mean reduction, rank-2 forward/reverse scans, rank-2/3
matrix multiplication including singleton-batch broadcasting, rank-≤4 strided
binary/unary/scalar elementwise operations, strided Kronecker products, and
matrix powers, matrix rank/determinant, strided dot/trace/L1/L2/max
map-reductions, and seeded random initializers. Sparse, streams, and storage
remain tracked follow-up families with differential CPU/WGPU contracts.

The hosted job checks out the sibling Atlas path repositories at their current
default branches. Those repositories are in an unpublished version migration,
so the job resolves the checkout-local path graph once before running the
verification commands with `--locked`. This is a temporary integration
constraint, not a dependency-resolution fallback; remove the bootstrap when
the sibling migration commits are published and the committed lockfile can
represent the hosted checkout graph directly.

## Implementation references

- [HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/): device
  acquisition, memory, transfer, synchronization, and attribute contracts.
- [`cubecl-hip-sys` 7.2.5321100](https://docs.rs/crate/cubecl-hip-sys/7.2.5321100):
  the Linux raw HIP bindings used by the optional feature.
- [ROCm Ubuntu development images](https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags):
  the pinned container source used by the container CI job; CI installs the
  smaller development tag's `rocm-hip-runtime-dev` package explicitly.
