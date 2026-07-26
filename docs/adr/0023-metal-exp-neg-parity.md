# ADR 0023 (hephaestus): Metal fused negated exponential parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

`hephaestus-core` defines `ExpNegOp` as the fused `exp(-x)` unary operation.
WGPU, CUDA, and ROCm export the marker. Metal's generic unary adapter already
accepts any WGPU `UnaryExpr`, but its public root omitted this marker, so a
Metal consumer could not select the same operation without reaching through a
different backend crate.

## Decision

Re-export `ExpNegOp` from the Metal elementwise module and crate root. Reuse the
existing generic unary dispatch through the native Metal-selected WGPU device;
do not add a Metal-specific shader or duplicate the WGPU operation family.

## Alternatives rejected

- Add a Metal-specific unary kernel: this duplicates an existing generic WGSL
  operation and creates a second implementation owner.
- Require consumers to compose `NegOp` and `ExpOp`: the fused operation is an
  existing public contract and composition changes the selected operation
  surface.
- Re-export the entire WGPU elementwise module: this would expose the WGPU
  provider boundary instead of a Metal-owned public surface.

## Verification

The Metal contract compares `ExpNegOp` with the CPU `(-x).exp()` oracle and the
`NegOp` then `ExpOp` composition. CUDA, ROCm, and macOS Metal provider CI run
the feature, warning-denied, Nextest, doctest, and rustdoc gates for the
resulting public surface.
