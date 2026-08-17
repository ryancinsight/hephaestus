# ADR 0051: Own accelerator 3D FDTD in Hephaestus

- Status: Accepted
- Date: 2026-08-17
- Board item: `HEPH-FDTD-PROVIDER-1`
- Change class: `[arch] [minor]`

## Context

Kwavers' FDTD validation needs a real provider execution path. The existing
consumer-owned WGPU code carries its own shader, buffer layout, and dispatch
ordering, while the solver-facing accelerator seam is f64-oriented and is not
connected to the equivalence runner. Hephaestus therefore has no canonical
contract against which a consumer can compare a CPU reference without either
duplicating provider logic or silently comparing CPU execution with itself.

## Decision

`hephaestus-core::domain::fdtd` owns one device-neutral collocated 3D FDTD
contract. `Fdtd3dParams` validates dimensions, spacings, timestep, and flattened
storage; `FdtdMedium` validates positive finite density and sound speed; and
`FdtdVelocity` fixes the typed four-lane storage layout. `Fdtd3dOps` prepares a
provider kernel and enqueues one in-place explicit-Euler step.

The contract is explicitly f32 because the initial WGPU implementation must
honor its native storage precision. The WGPU provider owns the prepared
velocity and pressure kernels, central-difference spacing, medium coefficients,
boundary handling, and velocity-before-pressure submission order. Consumers
own source injection, medium construction, CPU references, and equivalence
policy. No CPU fallback or f64 adapter belongs behind this seam.

## Alternatives

- Retain Kwavers' raw WGPU FDTD implementation: rejected because it duplicates
  provider ownership and leaves the consumer responsible for GPU dispatch.
- Reuse the f64 solver accelerator trait: rejected because it would require a
  consumer-side precision adapter and would not express typed provider buffers.
- Add a CPU fallback when WGPU acquisition fails: rejected because unavailable
  provider execution must remain an explicit typed failure.

## Verification

Core tests reject invalid geometry and medium values and validate buffer lengths.
The WGPU contract compares one provider step with an independent CPU
central-difference oracle for pressure and all velocity components. Local
compilation and focused Nextest pass; a device-required hosted WGPU run is the
merge gate. Kwavers must replace its raw consumer path and add the cross-repo
provider differential before this decision is fully integrated.

## Revisit trigger

Add another provider only when it implements the same typed seam and passes the
shared value contract. Revisit the f32 contract only with a provider-native
precision requirement and a numerical error analysis; do not widen and narrow
inside the existing operation.
