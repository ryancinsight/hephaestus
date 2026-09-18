# ADR 0061: Value semantics for kernel operators

- Status: Proposed
- Revision 2026-09-18 (second): a second review compiled the first revision's
  mechanism and found it unsound. A blanket
  `impl<Op: CombineValue> CombineExpr<Host> for Op` does not give a host seam
  impl, which receives only `Op: CombineExpr<Host>`, the `CombineValue` bound
  it needs to call `Op::combine` (E0277). A downstream type can implement
  `CombineExpr<Host>` with no value trait at all. The revision also left the
  Host `EXPR`/`TOKEN` strings and integer division undefined, and missed
  `EluOp`, `CeluOp` and the WGSL renderings in the correction list. Decisions
  3 to 6 and 8 below are rewritten; git holds the earlier text.
- Revision 2026-09-18 (first): the original draft made the value traits
  supertraits of the source traits, bound them on `RealField`, and counted
  six affected seams. That broke three implementors, excluded the integer
  scalars, and left the host dialect unresolvable.
- Date: 2026-09-18
- Refs: atlas `backlog.md#atlas-hephaestus-host-seam-coverage`; ADR 0046 §5
  (host implementors are admitted as clauses need them); atlas ADR 0038 (the
  conformance coverage table).

## Context

Seven clause families in `assert_backend_contract` are generic over an
operator bounded by a *kernel-source* trait: `FullReductionOps`,
`AxisReductionOps` and `ScanOps` over `CombineExpr`; `ElementwiseOps` over
`UnaryExpr`, `BinaryExpr` and `TypedBinaryExpr`; `TypedElementwiseBackend`
(`ElementwiseOps` over u32/i32/f32); `ParameterizedUnaryOps` over
`ParameterizedUnaryExpr`; and `StatefulUpdateOps` over the sealed
`StatefulUpdateRule`. The `*Expr` traits hold only `const EXPR: &str`, an
operator spelled in one dialect's source. A generating device needs exactly
that; an executing device, the host reference device of ADR 0046, has
nothing to apply. The fused seams (`fusion.rs`) sit outside the aggregate and
outside this decision.

Three constraints decide the design:

- The reduction and scan seams require `T: IdentityToken<Op,
  Self::Dialect>`, which requires `T: DialectScalar<Self::Dialect>`. The host
  must name a dialect, and borrowing `Wgsl` would inherit
  `IEEE_SPECIAL_VALUES = false`, which skips the special-value clause.
- The seams run over u32 and i32 as well as floats, so a floating-point bound
  on operator semantics excludes live instantiations.
- A seam impl cannot add bounds to the operator its method receives. Whatever
  the host calls must be reachable from `Op: CombineExpr<Host>` (and the other
  source traits) alone.

## Decision (recommended)

1. **Value traits, separate from the source traits.** Add `CombineValue`,
   `UnaryValue`, `BinaryValue`, `TypedBinaryValue<T>` and
   `ParameterizedUnaryValue` to `hephaestus-core`, each with one associated
   function applying the operator to scalars, implemented beside every
   operator marker's existing source impls. No supertrait relation, so
   existing implementors of the source traits (`hephaestus-wgpu`'s
   `SeamTypedExpr` adapter, test markers) are untouched. `StatefulUpdateRule`
   is sealed, so its value semantics are a method on that trait directly.
2. **Scalar bounds by operator family.** Combines, comparisons and integer-
   admitting binaries are generic over eunomia's `NumericElement` (u32, i32,
   f32, f64, F16, Bf16); real-valued unary and parameterized operators over
   `RealField`.
3. **Reaching the value from the source bound: one provided method per
   source trait.** Each source trait gains a provided method that can fail,
   for example on `CombineExpr<L>`:
   `fn value<T: NumericElement>(lhs: T, rhs: T) -> Option<T> { None }`, and on
   `UnaryExpr<L>`: `fn value<T: RealField>(x: T) -> Option<T> { None }`. A
   core-owned `Host: KernelDialect` (`IEEE_SPECIAL_VALUES = true`) carries
   blanket impls, `impl<Op: CombineValue> CombineExpr<Host> for Op`, that
   override the method with `Some(Op::combine(lhs, rhs))`. The host seam impl
   calls `<Op as CombineExpr<Host>>::value` under the bound it already
   receives. A type implementing a source trait for `Host` without a value
   trait inherits `None`, which the host reports as
   `HephaestusError::DispatchFailed` naming `core::any::type_name::<Op>()`.
   The method is provided, so adding it breaks no implementor.
4. **Host dialect constants.** `DialectScalar<Host>` for every scalar the
   seams accept; `IdentityToken<Op, Host>` blanket-implemented from
   `OpIdentity` (which exists for f32, u32 and i32, the scalars the clauses
   use). Host never renders source: its `EXPR` and `TOKEN` constants are the
   fixed string `"host"`. A generic blanket cannot compute an operator name in
   a `const` (`type_name` is not a stable `const fn`), so diagnostics use the
   runtime `type_name` instead.
5. **Integer paths.** `TypedElementwiseBackend` requires
   `ElementwiseOps<D, u32>` and `ElementwiseOps<D, i32>` in full, including
   unary operators whose value method is bound on `RealField`. The host
   dispatches per scalar through a host-local trait implemented for each
   seam scalar: float impls call the `RealField` value method, integer impls
   return the typed `DispatchFailed` error for real-valued unary operators.
   (A generic `impl<T: RealField>` beside concrete integer impls is rejected
   by coherence, since eunomia could later implement `RealField` for u32.)
   Integer arithmetic semantics follow WGSL, the one dialect that defines
   them: add, sub and mul wrap in two's complement; `x / 0` and
   `i32::MIN / -1` return `x`. CUDA and HIP renderings of signed combines are
   audited for reliance on signed overflow, which is undefined in C++; any
   that rely on it are rewritten to compute in the unsigned type and convert
   back.
6. **The value function is the operator's definition.** Where a kernel
   rendering diverges from it, the rendering is the defect. Five operators
   render with cancellation or overflow in all three dialects (WGSL, also
   used by Metal; CUDA C; HIP C):
   - `Expm1Op` renders `exp(x) - 1` and `Log1pOp` renders `log(1 + x)`:
     100% relative error as x approaches 0.
   - `EluOp` and `CeluOp` render `exp(x) - 1` on the negative branch: the
     same cancellation.
   - `SoftplusOp` renders `log(1 + exp(x))`: `inf` for x > 88 in f32 where
     the value is x.

   CUDA and HIP have `expm1`/`log1p` builtins. WGSL has none, so the WGSL
   renderings use cancellation-free formulations. For `log1p`, with
   `u = 1 + x`: `x` if `u == 1`, else `log(u) * x / (u - 1)`. For `expm1`,
   with `u = exp(x)`: `x` if `u == 1`, `-1` if `u - 1 == -1`, else
   `(u - 1) * x / log(u)`. For softplus: `max(x, 0) + log1p(exp(-|x|))`. The
   rendering item cites a resolved reference for each formulation and
   derives its WGSL tolerance. `MishOp` and `MishGradOp` embed the softplus
   rendering and move with it. The existing tests that pin the defective
   strings in `ops.rs` (for example the `Expm1Op`/`CudaC` assertion) change
   in the same item. Where a dialect has no exact builtin (WGSL `erf`,
   `erfc`, `lgamma`), the clause tolerance is derived from that
   approximation's documented error bound and stated per operator, never
   tuned.
7. **Match the renderings' semantics exactly.** `RoundOp` rounds half to even
   (WGSL `round`, CUDA `rint`), not eunomia's half-away-from-zero `round`;
   `SignOp` returns 0 for ±0 and NaN, not eunomia's `signum`. Gradient
   operators take their documented argument (`SigmoidGradOp`, `TanhGradOp`:
   the forward output; `ReluGradOp`, `GeluGradOp`: the input).
8. **eunomia first.** Eleven functions the value traits need are absent from
   eunomia: `asin`, `atan`, `acosh`, `asinh`, `atanh`, `exp2`, `expm1`,
   `log1p` on `RealField`, and `wrapping_add`, `wrapping_sub`,
   `wrapping_mul` on `NumericElement` (which has only `saturating_*` and
   `checked_*`). They are added there, per first-party supremacy, before the
   operators that use them.

## Alternatives

- **Value traits as supertraits of the source traits** (the original draft).
  Rejected: breaks every existing implementor of the source traits.
- **Blanket impls alone, with the host calling the value trait** (the first
  revision). Rejected: does not compile; the source bound cannot imply the
  value bound.
- **Add value bounds to the seam methods.** Rejected: breaking for every
  implementor and caller, repeated at every method.
- **Let the host borrow `Wgsl` as its dialect.** Rejected: disables the
  special-value clause on the one device where special values are exact.
- **An interpreter for `EXPR` strings.** Rejected: a second, string-typed
  definition of each operator.

## Consequences

- Additive: the new methods are provided and the new traits and dialect are
  new items, so no existing implementor or caller changes. Change class
  [minor][arch].
- `hephaestus-core` gains eunomia's `NumericElement`/`RealField` as bounds;
  today it uses only eunomia's `Pod`, `F16` and `Bf16`.
- All seven families become implementable on the host for the operators that
  carry value traits. Real-valued unary operators over integer scalars stay a
  typed error on the host.
- Five operator renderings across three dialects are corrected as a
  precondition, with their pinned tests.

## Verification plan

- A compile test shows a host-style impl calling
  `<Op as CombineExpr<Host>>::value` under only the source bound, and a
  downstream type implementing `CombineExpr<Host>` without a value trait,
  which yields `None`.
- Each value function is tested at special values (±0, ±inf, NaN,
  subnormals, integer wraparound, integer division by zero and
  `i32::MIN / -1`) and representative points against a direct reference.
- The corrected renderings are checked against the value functions on each
  available backend with derived tolerances.
- `hephaestus-host` then implements each seam and instantiates its existing
  conformance clause on the hosted runner.
