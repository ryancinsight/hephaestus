# ADR 0050: Product-axis reduction parity

- Status: Accepted
- Date: 2026-07-27
- Revision (2026-08-13): renumbered 0028 → 0050. It was drafted against a
  number already claimed by ADR 0028 (unary math-expression parity, landed
  2026-07-27), so it collided on landing 2026-07-30 and was never indexed.
  Content unchanged; the decision itself is not revisited.
- Scope: WGPU, CUDA, ROCm, and Metal rank-2 reduction application modules
- Change class: `[minor]`

## Context

PR #108 added `ProdOp` and allocated `prod_axis` wrappers to the four
providers. The public surface was incomplete: ROCm did not re-export the
shared marker, and the provider contracts did not pin caller-owned output,
both axes, non-contiguous input, empty-axis identity, or invalid output
layouts consistently. The implementation already has one shared core
identity/dialect vocabulary and one generic axis-reduction validation path per
native provider; the parity increment must close the contract surface without
adding a product-specific kernel.

## Decision

Expose `prod_axis` and `prod_axis_into` from every provider root and retain
`ProdOp` as the shared operation marker. WGPU, CUDA, and ROCm instantiate the
existing generic axis-reduction kernel for their dialect; Metal delegates the
same operation through the native Metal-selected WGPU device. The
multiplicative identity is `1` for `f32`, `u32`, and `i32` in host and shader
contracts. Empty reduced axes therefore write one for every surviving output
coordinate, while invalid axis, shape, storage, width, and alias contracts
continue to be owned by the shared axis planner.

## Rejected alternatives

- Add a product-specific kernel per provider: this duplicates the existing
  operation-parameterized reduction implementation and creates a second
  validation path.
- Implement product reduction in Coeus or Leto consumers: Hephaestus owns the
  accelerator dispatch and the public provider parity surface.
- Treat an empty axis as a no-op: the CPU contract defines the multiplicative
  identity, and leaving caller-owned output unchanged would make allocated and
  caller-owned forms observably inconsistent.

## Verification

Core expression tests pin `ProdOp` expressions, host identities, and identity
tokens for WGSL, CUDA C++, and HIP C++. Each provider contract compares
allocated and caller-owned product results on both axes against independent
closed-form fixture values, then covers a transposed layout, empty-axis
identity, and invalid output shape. The locked Leto release does not expose a
product-axis oracle; WGPU sum expectations retain the Leto differential path.
The local focused gates passed for core, WGPU, CUDA no-default, ROCm
no-default, and Metal no-default; the all-provider warning-denied clippy gate
and no-default doctests also passed. Default CUDA linking is blocked on this
Windows host by a missing `-lcuda`; hosted provider feature and hardware jobs
remain the merge evidence. Local rustdoc is additionally blocked before crate
compilation by a duplicate Moirai patch key in the neighboring Coeus manifest.

## Revisit trigger

Add prepared product-axis plans only when a consumer requires repeated product
dispatch and the prepared contract can retain the product-specific identity
resources without duplicating the immediate path.
