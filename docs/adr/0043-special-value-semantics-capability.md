# ADR 0043 — Special-value semantics as a dialect capability

- Status: Accepted
- Date: 2026-08-01
- Refs: atlas `backlog.md#atlas-arch-010`; meta ADR 0038 (capability-gated
  contract class); the removed `binary_elementwise_typed` NaN assertion that
  raised the item.

## Context

CUDA C++ and HIP C++ guarantee IEEE-754 semantics for `f32` arithmetic and
comparisons: `NaN != NaN` holds, NaN propagates through `+`/`*`, and
`Inf − Inf`/`0 × Inf` produce NaN. WGSL withholds that guarantee — the
specification permits implementations to "assume that overflow, infinities,
and NaNs are not present during shader execution", and an expression that
would produce one yields an indeterminate value. A shared conformance clause
asserting IEEE NaN ordering therefore failed on wgpu and was removed rather
than relaxed; but its removal left CUDA's and HIP's real, documented
semantics unverified. The gap is a contract that was never specified —
exactly the class meta ADR 0038 routes through a capability predicate.

## Decision

1. **The capability lives on the dialect, not the device.** Special-value
   behaviour is a property of the kernel language contract, so
   `KernelDialect` gains `const IEEE_SPECIAL_VALUES: bool`: `true` for
   `CudaC` and `HipC` (their language specifications promise IEEE-754
   `f32` special-value semantics), `false` for `Wgsl` (its specification
   withholds them). The const's Rustdoc is the single normative statement
   of what is and is not promised; seam traits reference it.
2. **The shared suite asserts IEEE semantics only where advertised.** The
   typed-elementwise conformance module gains a special-values clause
   compiled behind `S::Dialect::IEEE_SPECIAL_VALUES` (a const branch, so a
   non-advertising backend skips by construction, not by omission):
   unordered comparisons (`NaN == NaN` false, `NaN != NaN` true,
   `NaN < x` false), propagation (`NaN + 1`, `NaN × 1`), directed-infinity
   arithmetic (`Inf + 1 = Inf`), and NaN-producing indeterminate forms
   (`Inf − Inf`, `0 × Inf`). Oracles are exact bit-class checks
   (`is_nan`, `is_infinite` + sign), not epsilon comparisons.
3. **No blanket per-kernel non-finite rejection.** Kernels do not validate
   operand finiteness at dispatch: the check costs a full pass over hot
   inputs, and absence of a guarantee on WGSL is indeterminacy, not UB —
   the result is a wrong-but-safe value, the same failure class as any
   unconverged solve. Finiteness validation stays where
   `numerical_discipline` places it: at trust boundaries (validating
   newtypes, parser/FFI edges) and in solver termination checks, both of
   which run on the host. Consumers that need cross-backend determinism
   for non-finite inputs must not produce them on device paths — the
   dialect const is the queryable predicate for that decision.

## Alternatives

- Per-device capability (associated const on `ComputeDevice`): rejected —
  two devices sharing a dialect cannot disagree; placing it on the device
  invites drift between a device flag and the language it compiles.
- Uniform IEEE assertion with wgpu exempted in the test: rejected — an
  if-backend-name branch in a clause is exactly the by-omission skip the
  conformance crate exists to eliminate.
- Blanket device-side rejection of non-finite inputs: rejected per
  decision 3 (hot-path cost, wrong layer, and WGSL indeterminacy is not
  unsafety).

## Consequences

CUDA and HIP special-value semantics become verified contract; wgpu's
indeterminacy becomes documented contract instead of silence. Future
backends declare their position in one const. Kernels whose results feed
cross-backend differential tests must keep fixtures finite unless both
sides advertise the capability — the existing derived-tolerance rules
already assume finite values.
