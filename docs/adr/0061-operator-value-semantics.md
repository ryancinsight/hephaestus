# ADR 0061: Value semantics for kernel operators

- Status: Proposed
- Revision 2026-09-18 (fifth): a fifth review confirmed the binary dispatch
  shape by compilation and corrected Decision 6's measurements: the rejected
  regrouped forms' limit failures come from their infinity guard, SiLU
  underflows on [-91.86, -88.72], Softplus first overflows at 88.72, and
  Mish, MishGrad and Softplus also underflow in the negative tail. It also
  asked that Consequences and Verification cover every typed-error path, and
  that the rendering item exist before it is cited; it is
  [HEPH-KERNEL-TAIL-ACCURACY](../../backlog.md#heph-kernel-tail-accuracy).
- Revision 2026-09-18 (fourth): a fourth review measured the third
  revision's WGSL formulations over every f32 value: up to 2.5 ULP rather
  than the claimed 2 and 1, `expm1` returning x instead of +inf past 88.72,
  `log1p(-inf)` returning -inf, and a WGSL `log` accuracy bound (2^-21
  absolute) loose enough that the regrouped forms need not beat the current
  ones on a GPU. It also found the BinaryExpr dispatch rule sent f32 Add to a
  method Add does not override, and measured GeluTanh and Silu with the same
  underflow class. Decision 6 now fixes the principle and the defect list and
  hands formulation to its own item, where GPU measurement can settle it;
  Decisions 1 and 3 carry the corrected binary shape.
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
   function applying the operator to scalars (`BinaryValue` has a second,
   provided one; Decision 3), implemented beside every operator marker's
   existing source impls. No supertrait relation, so
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
   Binary operators split: Add, Sub, Mul and Div (every `BinaryExpr` marker
   but one) admit every `NumericElement`, while `PowOp` is defined only over
   reals. `BinaryValue` therefore has
   `fn apply<T: NumericElement>(lhs: T, rhs: T) -> Option<T>` and a provided
   `fn apply_real<T: RealField>(lhs: T, rhs: T) -> Option<T>` that defaults
   to `Self::apply` (legal because `RealField` implies `NumericElement`).
   Add implements only `apply`; Pow returns `None` from `apply` and overrides
   `apply_real`. `BinaryExpr<L>` mirrors the pair as provided `value` and
   `real_value`, both `None`, which the Host blanket overrides with the two
   `BinaryValue` methods. The host's per-scalar trait (Decision 5) calls
   `real_value` for f32 and f64 and `value` for every other scalar, so f32 Add
   reaches `apply` through the default and i32 Pow is the typed error. A
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
   seam scalar: f32 and f64 impls call the `RealField` value methods; u32,
   i32, F16 and Bf16 impls call the `NumericElement` methods and return the
   typed `DispatchFailed` error for real-only operators. eunomia implements
   `RealField` only for f32 and f64; implementing it for F16 and Bf16 is its
   own eunomia item, not assumed here.
   (A generic `impl<T: RealField>` beside concrete integer impls is rejected
   by coherence, since eunomia could later implement `RealField` for u32.)
   Integer arithmetic semantics follow WGSL, the one dialect that defines
   them: add, sub and mul wrap in two's complement; `x / 0` and
   `i32::MIN / -1` return `x`. Evidence: inferred from naga 30.0.1, the
   version hephaestus locks, whose HLSL and MSL writers implement exactly
   this division rule citing the WGSL specification (`back/hlsl/help.rs`,
   `back/msl/writer.rs`) and whose constant evaluator wraps i32/u32
   arithmetic citing the WGSL integer-types section; the item implementing
   the integer value functions confirms it against the specification text.
   CUDA and HIP renderings of signed combines are audited for reliance on
   signed overflow, which is undefined in C++; any that rely on it are
   rewritten to compute in the unsigned type and convert back.
6. **The value function is the operator's definition.** Where a kernel
   rendering diverges from it beyond the rendering's derived tolerance, the
   rendering is the defect, fixed in its dialect; the value function is
   never widened to match. Measured defects in f32 against f64 references,
   in WGSL (also Metal's dialect) and, except where noted, CUDA C and HIP C:
   - cancellation near zero: `Expm1Op` (`exp(x) - 1`), `Log1pOp`
     (`log(1 + x)`), and the negative branches of `EluOp` and `CeluOp`;
   - overflow: `SoftplusOp` (`log(1 + exp(x))`) is `inf` from x = 88.72,
     where the value is x;
   - tail underflow to zero where the value is a normal float: WGSL
     `ErfcOp` (`1 - erf(x)`); `GeluOp` and `GeluGradOp`
     (`1 + erf(x / sqrt(2))`); `GeluTanhOp` (-0 at x = -5.5 against
     -5.93e-9); `SiluOp` on [-91.86, -88.72]; and `SoftplusOp`, `MishOp` and
     `MishGradOp` in the negative tail (Softplus(-20) is 0 against 2.06e-9,
     Mish(-20) 0 against -4.12e-8, MishGrad 5% off there). `GeluTanhGradOp`
     and `SiluGradOp` share the forms and are measured by the rendering item.

   Choosing each replacement is the work of
   [HEPH-KERNEL-TAIL-ACCURACY](../../backlog.md#heph-kernel-tail-accuracy),
   not of this record: candidate forms must be measured on Vulkan, DX12 and
   Metal, because WGSL has no `isInf`, wgpu-hal's Metal path sets no math
   mode (`metal/device.rs:272`), so Apple's default, which may be fast-math
   and fold `(1 + x) - 1` to `x`, applies, and the WGSL `log` accuracy bound
   is reported as absolute (2^-21 on [0.5, 2]; the item confirms it against
   the specification's accuracy table). The item records the forms already
   rejected: the bare `log(u) * x / (u - 1)` and `(u - 1) * x / log(u)`
   overflow (`log1p(1e37)` and `expm1(85)` are `inf`) and `expm1(89)` is
   NaN where the value is `inf`; the regrouped `log(u) * (x / (u - 1))` and
   `(u - 1) * (x / log(u))` reach 2.47 ULP over all f32 values, and with a
   guard returning x when `u` is infinite they return x for `expm1` past
   88.72 (the value is `inf`) and -inf for `log1p(-inf)` (the value is NaN);
   the Abramowitz and Stegun 7.1.26 `erfc` is 3.5% off at x = 9 and its
   bound is absolute, so a clause using it cannot detect the tail defect. A
   tolerance is derived from each chosen form's published relative bound,
   never tuned, and the tests pinning today's strings in `ops.rs` change with
   the rendering.
7. **Match the renderings' semantics exactly.** `RoundOp` rounds half to even
   (WGSL `round`, CUDA `rint`), not eunomia's half-away-from-zero `round`;
   `SignOp` returns 0 for ±0 and NaN, not eunomia's `signum`. Gradient
   operators take their documented argument (`SigmoidGradOp`, `TanhGradOp`:
   the forward output; `ReluGradOp`, `GeluGradOp`: the input).
8. **eunomia first.** Thirteen functions the value traits need are absent
   from eunomia: `asin`, `atan`, `acosh`, `asinh`, `atanh`, `exp2`, `expm1`,
   `log1p` and a ties-to-even `round_ties_even` on `FloatElement`, where
   eunomia's existing transcendental functions live (its `round` rounds half
   away from zero); and `wrapping_add`, `wrapping_sub`, `wrapping_mul` and an
   integer division returning the dividend on a zero divisor or `MIN / -1`
   (floats keep IEEE division) on `NumericElement`, which has only
   `saturating_add`/`saturating_mul` and `checked_add`/`checked_mul`, so a
   generic `lhs / rhs` panics in Rust on those inputs. They are added there,
   per first-party supremacy, before the operators that use them.

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
  `Op: CombineExpr<L> + Other` calling `Op::value()` (or, for binaries,
  `Op::real_value()`) where `Other` also declares an associated item of that
  name becomes ambiguous (E0034), and a downstream
  `impl<L> IdentityToken<LocalOp, L> for f32` beside an
  `OpIdentity<LocalOp>` impl overlaps the Host blanket (E0119).
- `hephaestus-core` gains eunomia's `NumericElement`/`RealField` as bounds;
  today it uses only eunomia's `Pod`, `F16` and `Bf16`.
- All seven families become implementable on the host for the operators that
  carry value traits. Real-only operators (the real-valued unary and
  parameterized operators, and `PowOp`) stay a typed error on the host for
  every scalar without `RealField`: u32, i32, F16 and Bf16.
- The operator renderings in Decision 6 are corrected by their own item
  before the host clauses that exercise them land; the host's value
  functions do not wait on it.

## Verification plan

- A compile test shows a host-style impl calling
  `<Op as CombineExpr<Host>>::value` under only the source bound, and a
  downstream type implementing `CombineExpr<Host>` without a value trait,
  which yields `None`.
- A dispatch test shows f32 Add reaching `apply` through the `apply_real`
  default, Pow on f32 and f64 computing through `apply_real`, and Pow on i32
  and F16 returning the typed error.
- Each value function is tested at special values (±0, ±inf, NaN,
  subnormals, integer wraparound, integer division by zero and
  `i32::MIN / -1`) and representative points against a direct reference.
- The rendering item checks each corrected rendering against the value
  function on every available backend (Metal wherever a macOS host is
  available), with tolerances
  derived from the chosen forms' published relative bounds.
- `hephaestus-host` then implements each seam and instantiates its existing
  conformance clause on the hosted runner.
