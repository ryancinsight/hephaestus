# ADR 0071: Precision-generic stateful-update seam

- Status: Accepted
- Refs: #406 (debt-ratchet cure); atlas checker `type_suffixed_fns` 7 -> 0.

## Context

Stateful updates twinned every per-width item: `Parameters`/`ParametersF64`
structs, `update`/`update_f64` methods, five backend `stateful_update_f64`
entries, and a duplicated f64 conformance entry. The atlas checker counted
29 `type_suffixed_fns` sites in this family, while the f64 conformance entry
ran different fixture literals than the f32 entry around shared clause
bodies.

## Decision

One `UpdateWidth` seam (`Pod + FloatElement + Display`, plus a `Pad<N>` GAT:
f32 keeps the frozen uniform blocks, f64's pad is empty) carries every rule,
parameter struct, operation, and backend entry. `StatefulUpdateRule` exposes
a single `Parameters<T: UpdateWidth>: Pod` GAT plus one `update<T>` and one
`validate_parameters<T>`; the five parameter structs genericize as
`Name<T = f32>` so existing f32 call sites are unchanged. Layouts are
bit-identical to the deleted twins, pinned by per-width size asserts beside
the definitions.

Each backend fuses into one `stateful_update<Rule, T: UpdateWidth, N>`:
kernel text, parameter bytes, and pipeline-cache keys select the width at
compile time, while the dialect spelling crosses the `DialectScalar` bound
gap as a `scalar_token` value, rejecting unknown widths with a typed error
instead of a fallback spelling.

Conformance runs one generic `assert_stateful_update_contract<D, O, T>`
against a `StatefulUpdateWidth` clause seam (the `ContractScalar` pattern
from the convolution cure): fixture scalars lift through
`FloatElement::from_f32`, so f32 clauses run bit-identical values while f64
clauses widen the same f32-rounded fixtures, with backend and Leto oracle
always sharing one parameter value.

Eunomia's `Pod` derive rejects generic `repr(C)` (stable cannot prove the
layout padding-free), so the five parameter structs carry manual `unsafe
impl<T: UpdateWidth> Pod` blocks, each with a `// SAFETY` layout proof, a
reasoned `allow(unsafe_code)`, and adjacent size asserts; the crate is
otherwise `deny(unsafe_code)`.

## Consequences

BREAKING: the `*F64` parameter twins, `update_f64` twins, per-width backend
entries, and `assert_stateful_update_contract_f64` are deleted; callers use
the generic spellings (`SgdParameters::<f64>`,
`stateful_update::<Rule, f64, N>`, `assert_...::<_, _, f64>`). Workspace
minor bump to 0.20.0. Checker: `type_suffixed_fns` 7 -> 0,
`oversized_files` 36 -> 35, `orphan_modules` fixed; `allow_sites` rises
8 -> 13 for the five reasoned Pod allows (net debt down; the atlas baseline
regenerates after this lands). Tests pin eunomia `powi`'s squaring
evaluation (`b*(b*b)` for n = 3), which rounds 1 ulp off std's `powi` at
f32.
