# ADR 0072: Device-neutral n-D splitting seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `split` (slicing one tensor along an existing axis
into consecutive segments), the next remaining op after `CatOps`.

Like `cat`, `split` takes a runtime-count list — here outputs rather than
inputs. Coeus consumes it as `coeus_ops::split` (chunked forward split
with per-chunk `SplitNode` backward scatter), and its outputs are owned
tensors, so the seam materializes each segment into caller-owned storage
rather than returning zero-copy views; a zero-copy layout helper would not
serve the consumer, and one mechanism per op keeps the seam minimal.

## Decision

`SplitOps<D, T>` in `hephaestus-core::domain::split` provides `split_into`,
taking one `StridedView<D::Buffer<T>, N>`, an `axis`, `sizes: &[usize]`,
and caller-owned strided outputs. `validate_split_shape` checks the
geometry in a fixed order (axis range, sizes/outputs arity, per-output
shape agreement, checked axis summation) and returns the per-segment axis
bases, so backends never re-derive placement. No output buffer may alias
the input buffer or another output buffer; both backends reject aliasing
with `DispatchFailed` "must not alias" messages, and the shared clause
pins rejection-before-mutation for both the input-alias and
output-output-alias cases. Empty segments are legal and contribute
nothing, and empty `sizes` is legal exactly on a zero-length axis,
mirroring `leto::split`.

`WgpuSplitOps` dispatches one block-copy pass per output inside a single
encoder, all passes sharing one pipeline: a thread decodes its output
coordinate, reads the same coordinate with the `axis` lane shifted by the
validated base, and writes its own cell. No bounds check is needed
because validation proves each input window equals its output shape
exactly. The same binding-model reasoning as ADR 0071 applies (no
portable multi-buffer binding for a runtime output count), and each pass
likewise holds its pooled uniform buffer until the shared submit. The
host reference delegates to `leto::split` (ADR 0046) and scatters each
segment into its strided output, holding the input's read guard alive
through the copy loop because the segments borrow it.

CUDA, ROCm, and Metal do not implement `SplitOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`. Those backends
can dispatch single-launch with packed pointer arrays; the WGPU
per-output strategy is a backend-local choice the seam permits, not a
contract other backends must copy.

## Consequences

Coeus can bind `split_into` on WGPU (and the host reference pair) today.
`HEPH-SHAPE-OPS-PROVIDER-1` narrows further: stack, gather, scatter,
index_select, index_put, masked_fill, tile, where, sort, and nonzero
remain. CUDA/ROCm/Metal implementations and the aggregate fold are
follow-up items.
