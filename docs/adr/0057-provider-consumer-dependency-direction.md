# ADR 0057: Provider-consumer dependency direction

- Status: Proposed
- Date: 2026-09-04
- Board item: [`HEPH-WGPU-CONSUMER-2026-09-04`](../../backlog.md#heph-wgpu-consumer-2026-09-04)

## Context

`hephaestus-wgpu` is the Atlas WGPU provider, but it currently imports
`moirai-runtime` for blocking WGPU futures and `moirai-gpu` for launch planning
and the resource-budget type. That direction prevents a Moirai scheduling
adapter from consuming the provider and leaves two GPU ownership layers.

## Decision

Keep device acquisition, WGPU synchronization, typed buffers, and kernel
dispatch in Hephaestus. Replace provider use of the consumer runtime with a
provider-local, pure-Rust future executor, move covering-workgroup calculation
to the provider's existing `BlockWidth` contract, and remove the
`moirai-gpu` dependency. Hephaestus keeps `moirai-sync` only where its
provider-owned pooling implementation uses that independent substrate; it
does not import the Moirai runtime or GPU facade.

Downstream schedulers depend on `hephaestus-core` plus the concrete provider
and implement their own scheduling adapter. No compatibility dependency or
forwarding crate is retained.

## Alternatives rejected

- Retain the imports and add a reverse dependency from Moirai: Cargo cannot
  represent the resulting cycle, and provider ownership remains inverted.
- Copy the provider into Moirai: duplicates device semantics and violates the
  single-provider rule.
- Replace the provider with a generic WGPU facade: leaves the same dependency
  inversion and creates another seam.

## Verification

The provider's locked format, Clippy, unit, doctest, WGPU, CUDA, ROCm, and
cross-provider contract gates establish that replacing the blocking helper and
workgroup planner does not alter device behavior. A dependency-graph scan
asserts that `hephaestus-wgpu` no longer resolves either consumer package.
