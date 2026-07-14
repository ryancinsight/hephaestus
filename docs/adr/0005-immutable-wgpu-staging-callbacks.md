# ADR 0005: Immutable WGPU staging callbacks

Status: superseded by ADR 0006 and Mnemosyne ADR 0003 (2026-07-13)

## Context

Mnemosyne's WGPU staging backend now publishes one process-lifetime callback
pair instead of independently replaceable allocation and deallocation
functions. Hephaestus owns the concrete callbacks and the first WGPU device
whose mapped buffers back those allocations. Publishing the device before
registration could leave `HostPinned` allocation routed through another
provider's callbacks, and panicking inside either callback would unwind across
an FFI boundary.

## Decision

Hephaestus constructs one static `WgpuCallbacks` pair and registers it before
publishing the first staging device. `WgpuDevice::new` returns the existing
typed `Result`; a competing process registration becomes
`HephaestusError::DeviceUnavailable`, while repeated registration of the same
static pair is idempotent.

Both callbacks execute behind a shared panic boundary. Allocation converts a
panic or poisoned registry to null; deallocation converts either condition to
`false`. Normal allocation records the mapped buffer before returning its
pointer, and normal deallocation removes the same record before unmapping.

## Rejected alternatives

- Replacing a previously registered pair can mismatch live allocations and
  their deallocator.
- Ignoring a registration conflict would publish a staging device that the
  active callbacks do not own.
- Allowing lock-poison panics to propagate violates the callback pair's
  no-unwind contract.
- Aborting on callback failure removes the backend's existing fallible
  allocation contract without a correctness requirement for process abort.

## Failure modes and verification

Registration conflict is a typed construction error. Callback panics and
registry poison are allocation/deallocation failures. A value-semantic unit
test verifies the shared callback boundary returns the operation value on
success and the exact failure sentinel on panic. Mnemosyne race tests verify
one immutable pair wins concurrent registration; Hephaestus package checks,
clippy, nextest, doctests, and HostPinned contracts verify the consumer edge.
