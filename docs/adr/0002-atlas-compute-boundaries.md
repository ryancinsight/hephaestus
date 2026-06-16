# ADR 0002 (hephaestus): Atlas compute-boundary integration

## Status

Accepted.

## Context

Hephaestus must compose with the Atlas foundation crates without duplicating
their ownership domains:

- `mnemosyne-core` owns resource-budget vocabulary used by launch planning.
- `moirai-gpu` owns backend-agnostic GPU launch planning.
- `themis` owns topology snapshots for acquired accelerator devices.
- `hermes-simd` owns synchronous CPU SIMD over host slices.

Hermes explicitly does not own GPU resource lifetimes or device-resident
kernels; that boundary belongs to Hephaestus.

## Decision

Hephaestus integrates each crate at its owned layer:

- WGPU launch sizing constructs `KernelResourceBudget` and calls
  `moirai_gpu::plan_launch`, while preserving Hephaestus' checked
  `BlockWidth::checked_covering_blocks` overflow contract.
- WGPU and CUDA device acquisition expose `themis::GpuTopology` snapshots.
- Host-delegated dense linear-algebra wrappers call `leto-ops` with its `simd`
  feature enabled; Leto routes CPU hot loops through Hermes SIMD.
- Device-resident WGPU/CUDA kernels do not call Hermes directly. Direct Hermes
  calls from WGSL/CUDA kernels would cross the Atlas boundary because Hermes is
  a CPU SIMD substrate over host slices, not a GPU shader/PTX abstraction.

## Consequences

API parity wrappers that delegate factorization or matrix functions to Leto
use Hermes at the CPU tier and upload verified outputs into Hephaestus device
buffers. This is API and transfer-overhead parity, not GPU-kernel parity.

Native GPU kernels remain authored in the Hephaestus backend language
(WGSL/CUDA) and use Mnemosyne/Moirai/Themis where their domains apply. If a
future Hermes crate exposes a GPU-kernel IR or device-lane abstraction, this
ADR must be revised before Hephaestus consumes it.

Evidence tier: implementation audit against current dependencies and
value-semantic Hephaestus/Leto contract tests; no machine-checked proof.
