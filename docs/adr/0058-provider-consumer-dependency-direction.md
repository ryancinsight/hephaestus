# ADR 0058: Provider-consumer dependency direction

- Status: Accepted
- Date: 2026-09-04
- Board item: [`HEPH-WGPU-CONSUMER-2026-09-04`](../../backlog.md#heph-wgpu-consumer-2026-09-04)

Reviewed 2026-09-07 against `263f2d7`: independent source review accepts the
dependency direction and workgroup calculation. Locked workspace Clippy,
121 host tests, four doctests, and warning-denied documentation pass. Required
CUDA/WGPU execution passes 224 tests; ROCm host packing/source checks pass 33
tests. The bounded WGPU FFT smoke passes all 11 scenarios. HIP and Metal device
execution are not available on the Windows verification host.

## Context

Before this decision, the Atlas WGPU provider `hephaestus-wgpu` imported
`moirai-runtime` for blocking WGPU futures and `moirai-gpu` for launch planning
and the resource-budget type. That direction prevented a Moirai scheduling
adapter from consuming the provider and left two GPU ownership layers.

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

The provider's locked format, Clippy, unit, doctest, and device contract gates
check the blocking helper and workgroup planner. Device execution requires the
corresponding hardware; generated-source tests do not establish device behavior.

The 2026-09-07 locked dependency scan at `263f2d7` confirms that
`hephaestus-wgpu` has no direct `moirai-runtime` dependency and resolves no
`moirai-gpu` package. `leto-ops` still transitively selects `moirai-runtime`
0.5 with its async, iter, local, melinoe, and parallel features. It does not
activate GPU provider integration, so this retained CPU runtime path does not
restore the reverse GPU dependency. The oracle is `cargo tree --locked -p
hephaestus-wgpu -e features -i moirai-runtime`, alongside the provider manifest.
