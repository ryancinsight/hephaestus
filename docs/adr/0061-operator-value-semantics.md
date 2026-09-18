# ADR 0061: Value semantics for kernel operators

- Status: Proposed
- Revision 2026-09-18 (third): a third review compiled Decisions 3 and 5 and
  confirmed them, but found the WGSL `log1p`/`expm1` formulations overflow
  (`log(u) * x` is `inf` at `log1p(1e37)` in f32), `ErfcOp` and the Gelu
  renderings carry the same cancellation class unlisted, `BinaryExpr`'s value
  bound was unstated (`PowOp` is real-only), `RealField` covers only f32/f64,
  and the eunomia list lacked integer division and ties-to-even rounding.
  Decisions 3, 5, 6 and 8 and Consequences are amended.
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
   `UnaryExpr<L>`: `fn value<T: RealField>(x: T) -> Option<T> { None }`.
   `BinaryExpr<L>` carries both an integer-admitting
   `fn value<T: NumericElement>` and a `fn real_value<T: RealField>`, because
   its operators split: Add, Sub, Mul, Div, Min and Max admit every
   `NumericElement`, while `PowOp` is defined only over reals; each operator
   overrides the methods its value trait supports, and the host's per-scalar
   trait (Decision 5) selects which one it calls. A
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
   seam scalar: f32 and f64 impls call the `RealField` value methods; u32 and
   i32 impls call the `NumericElement` methods and return the typed
   `DispatchFailed` error for real-valued operators. eunomia implements
   `RealField` only for f32 and f64, so the F16 and Bf16 impls return the same
   typed error for real-valued operators until eunomia implements `RealField`
   for them, which is filed as its own eunomia item rather than assumed here.
   (A generic `impl<T: RealField>` beside concrete integer impls is rejected
   by coherence, since eunomia could later implement `RealField` for u32.)
   Integer arithmetic semantics follow WGSL, the one dialect that defines
   them: add, sub and mul wrap in two's complement; `x / 0` and
   `i32::MIN / -1` return `x`. Evidence: inferred from naga 30.0.1, the
   version hephaestus locks, whose HLSL and MSL writers implement exactly
   this division rule citing the WGSL specification (`back/hlsl/help.rs`,
   `back/msl/writer.rs`) and whose constant evaluator wraps i32/u32
   arithmetic citing the WGSL integer-types section; the rendering item
   confirms it against the specification text. CUDA and HIP renderings of signed combines are
   audited for reliance on signed overflow, which is undefined in C++; any
   that rely on it are rewritten to compute in the unsigned type and convert
   back.
6. **The value function is the operator's definition.** Where a kernel
   rendering diverges from it, the rendering is the defect. Eight operators
   render with cancellation or overflow (WGSL, also used by Metal; CUDA C;
   HIP C):
   - `Expm1Op` renders `exp(x) - 1` and `Log1pOp` renders `log(1 + x)`:
     100% relative error as x approaches 0.
   - `EluOp` and `CeluOp` render `exp(x) - 1` on the negative branch: the
     same cancellation.
   - `SoftplusOp` renders `log(1 + exp(x))`: `inf` for x > 88 in f32 where
     the value is x.
   - WGSL `ErfcOp` renders `1 - erf(x)` through the polynomial `erf`: 0 for
     x above about 4, where `erfc` is small but positive.
   - `GeluOp` and `GeluGradOp` render `1 + erf(x / sqrt(2))` in all three
     dialects: 0 for x below about -6. The cancellation-free form is
     `erfc(-x / sqrt(2))`, which moves with the `ErfcOp` fix in WGSL.

   CUDA and HIP have `expm1`/`log1p` builtins. WGSL has none, so the WGSL
   renderings use cancellation-free formulations, grouped so no intermediate
   overflows. For `log1p`, with `u = 1 + x`: `x` if `u == 1` or `u` is
   infinite, else `log(u) * (x / (u - 1))`. For `expm1`, with `u = exp(x)`:
   `x` if `u == 1` or `u` is infinite, `-1` if `u - 1 == -1`, else
   `(u - 1) * (x / log(u))`. The ungrouped products `log(u) * x` and
   `(u - 1) * x` overflow (`log1p(1e37)` and `expm1(85)` become `inf` in f32)
   and are rejected. A host f32 sweep against f64 references puts the grouped
   forms within 2 and 1 ULP, with every limit case exact. For softplus:
   `max(x, 0) + log1p(exp(-|x|))`. The rendering item cites a resolved
   reference for each formulation, derives its WGSL tolerance, and runs on
   Metal as well as Vulkan and DX12: wgpu-hal's Metal path keeps Apple's
   default math mode, which may be fast-math and fold `(1 + x) - 1` to `x`,
   undoing both formulations. `MishOp` and `MishGradOp` embed the softplus
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
8. **eunomia first.** Thirteen functions the value traits need are absent
   from eunomia: `asin`, `atan`, `acosh`, `asinh`, `atanh`, `exp2`, `expm1`,
   `log1p` and a ties-to-even `round_ties_even` on `FloatElement`, where
   eunomia's existing transcendental functions live (its `round` rounds half
   away from zero); and `wrapping_add`, `wrapping_sub`, `wrapping_mul` and a
   division returning the dividend on a zero divisor or `MIN / -1` on
   `NumericElement` (which has only `saturating_add`/`saturating_mul` and
   `checked_add`/`checked_mul`, so a generic `lhs / rhs` panics in Rust on
   those inputs). They are added there, per first-party supremacy, before the
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
  [minor][arch]. Two patterns that compile today would break, and a search of
  every registered stack member at origin finds neither: a generic
  `Op: CombineExpr<L> + Other` calling `Op::value()` where `Other` also
  declares an associated `value` becomes ambiguous (E0034), and a downstream
  `impl<L> IdentityToken<LocalOp, L> for f32` beside an
  `OpIdentity<LocalOp>` impl overlaps the Host blanket (E0119).
- `hephaestus-core` gains eunomia's `NumericElement`/`RealField` as bounds;
  today it uses only eunomia's `Pod`, `F16` and `Bf16`.
- All seven families become implementable on the host for the operators that
  carry value traits. Real-valued unary operators over integer scalars stay a
  typed error on the host.
- Eight operator renderings across three dialects are corrected as a
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
