# ADR 0061: Value semantics for kernel operators

- Status: Proposed
- Revision 2026-09-18: rewritten after an independent review against the
  code (compile experiments on cargo 1.97). The first draft made the value
  traits supertraits of the source traits, bound them on `RealField`, and
  counted six affected seams. That breaks three implementors (one in
  production wgpu code), excludes the integer scalars the seams accept, leaves
  the host's `type Dialect` unresolvable, and misses two families. Superseded
  in place; git holds the draft.
- Date: 2026-09-18
- Refs: atlas `backlog.md#atlas-hephaestus-host-seam-coverage`; ADR 0046 §5
  (host implementors are admitted as clauses need them); ADR 0041 (the
  conformance crate and its coverage table).

## Context

Seven clause families in `assert_backend_contract` are generic over an
operator bounded by a *kernel-source* trait: `FullReductionOps`,
`AxisReductionOps` and `ScanOps` over `CombineExpr`; `ElementwiseOps` over
`UnaryExpr`, `BinaryExpr` and `TypedBinaryExpr`; `TypedElementwiseBackend`
(`ElementwiseOps` over u32/i32/f32); `ParameterizedUnaryOps` over
`ParameterizedUnaryExpr`; and `StatefulUpdateOps` over the sealed
`StatefulUpdateRule`. The `*Expr` traits hold only `const EXPR: &str`, an
operator spelled in one dialect's source. A generating device needs exactly
that; an executing device — the host reference device of ADR 0046 — has
nothing to apply.

Two further constraints decide the design:

- The reduction and scan seams also require `T: IdentityToken<Op,
  Self::Dialect>`, which requires `T: DialectScalar<Self::Dialect>`. The host
  must therefore name a dialect, and borrowing `Wgsl` would inherit
  `IEEE_SPECIAL_VALUES = false`, which skips the special-value clause.
- The seams run over u32 and i32 as well as floats (`OpIdentity` impls exist
  for both), so a floating-point bound on operator semantics excludes live
  instantiations.

## Decision (recommended)

1. **Value traits, separate from the source traits.** Add `CombineValue`,
   `UnaryValue`, `BinaryValue`, `TypedBinaryValue<T>` and
   `ParameterizedUnaryValue` to `hephaestus-core`, each with one associated
   function applying the operator to scalars, implemented beside every
   operator marker's existing source impls. No supertrait relation: existing
   implementors of the source traits — including
   `hephaestus-wgpu`'s `SeamTypedExpr` adapter and test markers — are
   untouched. `StatefulUpdateRule` is sealed, so its value semantics are a
   method on that trait directly.
2. **Scalar bounds by operator family.** Combines, comparisons and integer-
   admitting binaries are generic over eunomia's `NumericElement` (u32, i32,
   f32, f64, F16, Bf16); real-valued unary operators over `RealField`. Integer
   combines wrap, matching the WGSL and CUDA kernels — stated in each impl.
3. **A core-owned `Host` dialect.** `Host: KernelDialect` with
   `IEEE_SPECIAL_VALUES = true`, `DialectScalar<Host>` for every scalar the
   seams accept, `IdentityToken<Op, Host>` from `OpIdentity`, and blanket
   impls of each source trait for `Host` from the matching value trait. Host
   never renders source; its `EXPR` constants are the operator's name, used
   only in diagnostics, and a test asserts no host code path compiles a
   kernel. `hephaestus-host`'s seam impls set `type Dialect = Host` and apply
   `Op::combine`/`Op::apply` under bounds they already receive.
4. **The value function is the operator's definition.** Where a kernel
   rendering diverges from it, the rendering is the defect: `Expm1Op` and
   `Log1pOp` render `exp(x) - 1` and `log(1 + x)` (100% relative error near
   0; CUDA and HIP have `expm1`/`log1p`), and `SoftplusOp` renders
   `log(1 + exp(x))` (overflows for x > 88 in f32). Those renderings are fixed
   before their host counterparts land. Where a dialect has no exact builtin
   (WGSL `erf`, `erfc`, `lgamma` are polynomial approximations), the clause
   tolerance is derived from that approximation's documented error bound and
   stated per operator — never tuned.
5. **Match the renderings' semantics exactly.** `RoundOp` rounds half to even
   (WGSL `round`, CUDA `rint`), not eunomia's half-away-from-zero `round`;
   `SignOp` returns 0 for ±0 and NaN, not eunomia's `signum`. Gradient
   operators take their documented argument (`SigmoidGradOp`, `TanhGradOp`:
   the forward output; `ReluGradOp`, `GeluGradOp`: the input).
6. **eunomia first.** Eight functions the operators need are absent from
   eunomia under any name — `asin`, `atan`, `acosh`, `asinh`, `atanh`,
   `exp2`, `expm1`, `log1p` — and are added there, per first-party supremacy,
   before the operators that use them.

## Alternatives

- **Value traits as supertraits of the source traits** (the first draft).
  Rejected: breaks every existing implementor of the source traits, and
  `TypedBinaryExpr`'s wgpu adapter cannot be repaired without a value trait
  the draft did not have.
- **Add value bounds to the seam methods.** Rejected: breaking for every
  implementor and caller, repeated at every method.
- **Let the host borrow `Wgsl` as its dialect.** Rejected: disables the
  special-value clause on the one device where special values are exact.
- **An interpreter for `EXPR` strings.** Rejected: a second, string-typed
  definition of each operator.

## Consequences

- Additive: no existing implementor or caller changes. Change class
  [minor][arch].
- `hephaestus-core` gains eunomia's `NumericElement`/`RealField` as bounds;
  today it uses only eunomia's `Pod`, `F16` and `Bf16`.
- All seven families become implementable on the host, and ADR 0041's
  coverage table can record every clause running on a GPU-less runner.
- Three shipped kernel renderings are corrected as a precondition.

## Verification plan

Per family: each value function is tested at special values (±0, ±inf, NaN,
subnormals, integer wraparound) and representative points against a direct
reference; the corrected renderings are checked against the value functions
on each available backend with derived tolerances; then `hephaestus-host`
implements the seam and instantiates its existing conformance clause on the
hosted runner.
