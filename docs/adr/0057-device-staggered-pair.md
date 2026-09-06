# ADR 0057: The device staggered pair gathers a derived transpose

- Status: Accepted
- Date: 2026-09-04
- Item: `backlog.md#heph-staggered-3d-2026-09-04`
- Related: [ADR 0002](0002-atlas-compute-boundaries.md) (what hephaestus-core
  may depend on), atlas `docs/audit/math-ssot-ledger.md`, kwavers ADR 128

## Context

Leto owns the CPU staggered gradient/divergence pair as
`leto_ops::StaggeredLeapfrog3D`, and Coeus binds Leto and Hephaestus as backends
of one seam. Until now Hephaestus exposed only a 2-D Laplacian, so a consumer's
FDTD sweep could reach a CPU implementation and nothing else. This record covers
the three decisions the device half forced.

## Decision 1: a separate trait, not new methods on `StencilOps`

`Staggered3DOps<D>` is its own seam. Adding the methods to `StencilOps` would
have obliged every backend to supply bodies before it had kernels, and a body
that returns zeros or an error is a mock wearing a trait impl. A backend
advertises the staggered capability exactly when it has one; consumers bind
whichever seam they need.

WGPU implements it, and Metal by delegation to the WGPU kernel — the same
arrangement the 2-D Laplacian already uses. CUDA and ROCm follow separately.

## Decision 2: the divergence gathers a hand-derived transpose

Leto computes the divergence by scattering `-Gᵀ` directly. That is why its
adjoint identity is true by construction: the transpose is applied rather than
re-expressed, so the wall closure cannot disagree with the gradient's.

A GPU cannot scatter without atomics, so the device kernel gathers, which means
writing the transpose out as its own stencil — including the wall closure the
Leto comment names as the thing that is easy to get wrong.

It is derived rather than guessed. For
`G[i] = Σ_n c_n (f[R(i+n)] − f[R(i−n+1)])` the transpose's column `j` collects
every `i` whose tap lands on `j`:

```text
  D[j] = Σ_n c_n [ A_n(j) − B_n(j) ]
  A_n(j) = u[j+n−1]  when j+n−1 < extent      (unreflected preimage)
         + u[n−2−j]  when j ≤ n−2             (reflected across the low wall)
  B_n(j) = u[j−n]              when j ≥ n
         + u[2·extent−1−j−n]   when j+n ≥ extent  (reflected across the high wall)
```

The derivation is only half of it. The claim is checked three ways, because a
derivation and its transcription into WGSL are different things:

1. against the CPU pair, cell by cell, on every axis at orders 2, 4, 6, and 8;
2. by the adjoint identity computed on the device's own outputs, which a
   mistake shared between the two operators could still break; and
3. by a constant field, whose gradient is exactly zero everywhere only if the
   walls reflect rather than zero-extend.

The suite was shown to bite rather than assumed to: flipping the sign of the
low-wall reflected term failed exactly the three cases that should catch it.

## Decision 3: the taps are a parameter, not derived in core

An order-`2N` staggered stencil's coefficients come from solving a Taylor
system. Deriving them per dispatch would put a linear solve inside a timestep,
so they are derived once and ride in the uniform block.

They are derived by the *caller* rather than by `hephaestus-core`, because that
crate carries Leto's layout vocabulary and no CPU compute dependency (ADR 0002,
atlas ADR 0001) — a linear solve there is precisely the dependency the boundary
exists to exclude. Callers pass the output of
`leto_ops::staggered_first_derivative_coefficients`, the same function the CPU
path uses.

That parameter opens a gap: nothing in the type system says the taps came from
the provider. The conformance clauses close it behaviourally — a device whose
taps are not the provider's fails the ramp and adjoint clauses.

## Consequence: a documented capability difference

Leto's reflection loops, so a stencil deeper than its axis still lands in range.
Under `extent >= 2N` a single reflection step is exact, which is what keeps the
device kernels loop-free in their reflection. `Staggered3DParams::new` enforces
that precondition and rejects thinner grids with a typed error naming the CPU
backend as the path that serves them.

This is a rejected configuration, not a silent divergence, and it is stated here
because a backend seam whose two sides quietly disagree on the edges is worse
than one that admits an edge it does not serve.

## What remains

**Revision 2026-09-06 — CUDA implemented.** `CudaStaggered3DOps` lands the same
three decisions in CUDA C, verified on a live device (RTX 5080, CUDA 13.3) by
the differential against the CPU pair at orders 2/4/6/8 on every axis, the
device-side adjoint identity, and the shared conformance clauses — with the
sign-flip mutation check repeated to show the differential bites. The decisions
above are unchanged; only the set of backends implementing them grew.

ROCm remains. Its kernel source would be byte-identical to CUDA's — plain
CUDA/HIP C, no vendor intrinsics — so it is the second consumer and the source
consolidates to one shared home rather than being pasted into a second crate;
the existing per-backend Laplacian sources are the precedent to avoid. It is
parked because the development host has no AMD device, and the differential
against the CPU pair is the only oracle that makes a hand-derived gathered
transpose trustworthy: landing an unverifiable copy is precisely what the
mutation check exists to prevent. Until it lands, ROCm does not implement
`Staggered3DOps` — the honest state, and the reason this is a separate trait.
