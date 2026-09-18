# ADR 0061: Value semantics for kernel operators

- Status: Proposed
- Date: 2026-09-18
- Refs: atlas `backlog.md#atlas-hephaestus-host-seam-coverage`; ADR 0046 §5
  (host implementors are admitted as clauses need them); ADR 0041 (the
  conformance crate and its coverage table).

## Context

Six compute seams are generic over an operator marker bounded by a
*kernel-source* trait: `FullReductionOps`, `AxisReductionOps` and `ScanOps`
over `Op: CombineExpr<Self::Dialect>`; `ElementwiseOps`,
`ParameterizedUnaryOps` and `StatefulUpdateOps` over `UnaryExpr`/`BinaryExpr`
and their kin. Each of those traits carries exactly one thing — `const EXPR:
&str`, the operator spelled in a dialect's source language. That is the right
contract for a device that generates kernels, and it is the only contract the
seam offers.

A device that executes instead of generating — the host reference device of
ADR 0046 — receives an `Op` it cannot apply: it can read the expression text
but not evaluate it, and a trait impl cannot add bounds to a trait method, so
it cannot ask for anything else. The six seams are therefore unimplementable
on the host as defined, and their conformance clauses run only on machines
with the matching accelerator. On a GPU-less hosted runner they never run.

## Decision (recommended)

1. **Operators carry their value semantics, independent of dialect.** Add
   `CombineValue`, `UnaryValue` and `BinaryValue` traits to `ops.rs`, each with
   one associated function applying the operator to scalars
   (`fn combine<T: RealField>(lhs: T, rhs: T) -> T`, and likewise), and
   implement them for every operator marker beside its existing `*Expr` impls.
2. **They are supertraits of the source traits.** `CombineExpr<L>:
   CombineValue`, `UnaryExpr<L>: UnaryValue`, `BinaryExpr<L>: BinaryValue`.
   Every existing bound `Op: CombineExpr<Self::Dialect>` then already implies
   the value semantics, so no seam signature changes and no caller adds a
   bound; an executing implementor calls `Op::combine(a, b)` under the bound it
   was given.
3. **Scalar math comes from eunomia.** The value functions are generic over
   eunomia's `RealField`, the stack's owner of scalar arithmetic. Operators
   whose function eunomia lacks (`expm1`, `log1p`, `asinh`, `atanh` were not
   found under those names on 2026-09-18) are added to eunomia first, per
   first-party supremacy — not approximated here.
4. **One definition of each operator.** An operator's source expression and
   its value function are two renderings of one mathematical definition, so a
   shared differential test evaluates each value function against a direct
   reference at special values (±0, ±inf, NaN, subnormals) and representative
   points; the device conformance clauses then check every backend's kernel
   against the same values.

## Alternatives

- **Add `Op: CombineValue<T>` to the six seams' method bounds.** Rejected: a
  breaking change to every generic caller for a capability the operator
  already has, and the bound would be repeated at every seam method.
- **A host dialect whose `EXPR` the host interprets.** Rejected: an
  expression interpreter is a second, string-typed definition of each
  operator — the duplication this ADR removes, with parse failures at run
  time instead of compile errors.
- **Host implementations only for the eleven value-shaped seams.** Leaves six
  clause families unverifiable in CI indefinitely; recorded as the state
  until this lands, not as an outcome.

## Consequences

- Every operator marker gains one small impl; a new operator cannot compile
  without its value semantics, so the gap cannot reopen.
- The supertrait makes `hephaestus-core` depend on eunomia's `RealField`,
  which it already consumes for scalar types.
- `hephaestus-host` can implement all six seams, and ADR 0041's coverage
  table can record every clause running on the host.

## Verification plan

Per operator family: the value-function differential test above, then the
host implementation of each seam instantiating that seam's existing
conformance clause in `hephaestus-host`'s tests, run on the hosted runner.
