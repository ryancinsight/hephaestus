# ADR 0028: Unary math expression parity

- Status: accepted
- Date: 2026-07-27
- Scope: Hephaestus unary expression vocabulary and WGPU, CUDA, ROCm, and
  Metal elementwise providers
- Change class: `[minor]`

## Context

Leto's CPU unary operation vocabulary includes common trigonometric,
hyperbolic, logarithmic, exponential, sign, and rounding operations. Coeus
already has WGPU and CUDA shader forms for these operations, but Hephaestus
only exposed the smaller set used by the existing activation and fused
expression paths. ROCm and Metal therefore could not consume one shared
operation vocabulary for this part of the Coeus contract.

The current consumer contract is f32. The expression seam is intentionally
dialect-specific because WGSL uses unsuffixed literals and `select`, while
CUDA C++ and HIP C++ use f-suffixed literals, conditional expressions, and
`rint` for round-to-nearest-even.

## Decision

Add one zero-sized marker for each of the following operations:

`tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `log2`, `log10`, `exp2`,
`atanh`, `asinh`, `acosh`, `expm1`, `log1p`, `sign`, `floor`, `ceil`, `round`,
and `trunc`.

Implement the WGSL and CUDA C++ forms through one macro and add the HIP C++
forms to the existing HIP expression table. Re-export the same markers from
all four backend application modules. Coeus routes these markers only for its
f32 activation-capable provider path; integer arithmetic-only paths continue
to reject transcendental and rounding operations with their typed unsupported
operation error.

`log10` and `expm1` use the same composed forms already present in the Coeus
WGSL contract. `sign` preserves the zero result for signed zero and ordinary
zero inputs. CUDA and HIP `round` use `rint`, matching the existing Coeus
CUDA contract's ties-to-even behavior.

## Alternatives rejected

- Copy the operation match arms into each Coeus backend: rejected because it
  duplicates the provider vocabulary and leaves the Hephaestus seam unused.
- Add a CPU fallback for ROCm or Metal: rejected because it would hide a
  missing provider implementation and violate the backend ownership boundary.
- Add `erf`, `erfc`, or `lgamma` in this increment: rejected because WGPU does
  not currently provide a common native expression for them; exact parity
  requires a separate shared approximation or provider capability decision.
- Add f64 markers now: rejected because the active Coeus provider contract is
  f32 and the current Hephaestus unary seam is not scalar-parameterized.

## Verification

Core tests pin representative direct expressions, composed forms, sign
selection, and rounding dialect differences. Coeus ROCm and Metal tests use
the Leto CPU unary oracle across valid input domains for each operation. The
four backend workflows are the authoritative hosted compile and contract
check; required physical-device jobs remain separate evidence from
adapterless compilation and are reported independently.

## Revisit trigger

Revisit this decision when Coeus requires f64, reduced-precision, vector, or
additional unary operations, or when a common native approximation for
`erf`/`erfc`/`lgamma` can be specified and differentially verified across all
four backends.
