# ADR 0081: Device-neutral n-D conditional-selection seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26); ADR 0043 (IEEE special
  values per dialect).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `where` (the element-wise ternary selecting
`on_true` under a nonzero condition and `on_false` elsewhere), the next
remaining op after `TileOps`. It is the same-shape contract
`coeus_ops::where_cond` documents, which today composes the selection
from seven backend-primitive dispatches with intermediate temporaries —
two `ReluGrad` masks, a negation, an add, a `full`, a subtraction, two
multiplies, and a final add — while this seam runs it as one kernel with
no temporaries.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. Like `MaskedFillOps`,
this seam interprets `T` (a zero comparison with non-finite edges)
rather than copying it, so it carries the same `type Dialect:
KernelDialect` and the same ADR 0043 const-branch structure: finite
conditions pin everywhere, and a `NaN` condition selects `on_true` only
where the dialect advertises IEEE special values.

## Decision

`WhereOps<D, T>` in `hephaestus-core::domain::where_cond` provides
`where_into`, taking a strided condition, strided `on_true` and
`on_false` branches, and a caller-owned strided output. All four shapes
must be equal — no broadcasting — checked by `validate_where_shape` in a
fixed order (`on_true`, `on_false`, then output, each against the
condition). The output buffer must alias none of the three source
buffers, and the shared clause pins rejection-before-mutation for each.
Empty shapes select vacuously.

The condition test is scalar inequality against zero, pinned for every
finite value on every backend: any nonzero magnitude selects `on_true`,
and negative zero selects `on_false`. A `NaN` condition selects
`on_true` only on IEEE-advertising dialects, asserted behind the same
const branch the masked-fill suite uses and skipped by construction on
WGSL. (The module is named `where_cond` because `where` is a Rust
keyword.)

`WgpuWhereOps` dispatches one thread per output cell in a single launch:
decode the coordinate, read the condition and both branch cells, and
write the selected branch. Its four storage buffers exactly fill the
portable `max_storage_buffers_per_shader_stage` limit of four, so the
single-launch shape is also the only portable one. There is no error
flag — the op takes no indices, so no data-dependent fault exists. The
host reference walks the condition through `leto::ArrayView` and tests
against `eunomia::NumericElement::ZERO`, keeping `-0.0`/`NaN` semantics
in Rust's IEEE arithmetic rather than a bitwise test.

CUDA, ROCm, and Metal do not implement `WhereOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `where_into` on WGPU (and the host reference pair) today,
retiring the seven-dispatch composition for the covered backends.
`HEPH-SHAPE-OPS-PROVIDER-1` narrows further: sort and nonzero remain.
CUDA/ROCm/Metal implementations and the aggregate fold are follow-up
items; the CUDA and ROCm residuals will exercise the `NaN`-selects-
`on_true` cell their dialects advertise.
