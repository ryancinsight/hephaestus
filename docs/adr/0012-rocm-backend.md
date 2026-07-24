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
consumer acceptance oracle: contiguous binary, unary, and scalar elementwise
operations, contiguous and rank-2 axis reductions, and rank-2 scans can be
compared against CPU values and the existing CUDA/WGPU contracts.

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
module entry point per `(operation, scalar, block width)` key. Output buffers
must be distinct from inputs, matching the CUDA/WGPU elementwise contract.
Contiguous reductions use the same cached module shape for each tree pass and
retain typed partial buffers until the final one-element result is returned;
empty inputs return the typed operation identity. Rank-2 axis reductions use
the shared `AxisReductionMeta` and `plan_axis_reduction` contracts, so shape,
stride, storage, alias, and empty-axis validation remains provider-neutral.
Rank-2 scans use the shared `AxisScanMeta` and `plan_axis_scan` contracts. One
HIP block owns each logical scan line, lanes fold contiguous chunks in order,
and shared-memory chunk prefixes complete forward or reverse cumulative sums
and products without a second provider-specific host planner.
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
axis sum/min/max/mean reduction, and rank-2 forward/reverse scans. Linalg,
sparse, strided elementwise, streams, storage, and random operations remain
tracked follow-up families with differential CPU/WGPU contracts.

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
