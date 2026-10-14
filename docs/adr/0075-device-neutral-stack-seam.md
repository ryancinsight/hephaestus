# ADR 0075: Device-neutral n-D stacking seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

- Revised 2026-10-14: the CUDA/ROCm/Metal "not implemented yet" paragraphs below
  are stale 
—
 those backends have since landed (rescue slices 3, 4, 11a). Other
  follow-ups keep their stated status.

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `stack` (stacking equal-shaped tensors along a new
axis), the next remaining op after `SplitOps`.

Unlike every other seam in this family, `stack` changes rank: rank-`N`
inputs produce a rank-`M = N + 1` output. Coeus consumes it as
`coeus_ops::stack` (forward stack with per-input `StackNode` backward
select), and its output is an owned tensor, so the seam materializes the
stack into caller-owned storage. The new axis ranges over `0..=N`: a new
trailing axis is legal, unlike concatenation's `axis < N`.

## Decision

`StackOps<D, T>` in `hephaestus-core::domain::stack` provides
`stack_into<const N: usize, const M: usize>`, taking
`&[StridedView<D::Buffer<T>, N>]`, an `axis`, and a caller-owned
`StridedView<D::Buffer<T>, M>`. The method carries leto's own
`RankMarker<N>: InsertAxis<N, LargerShape = [usize; M]>` bound, so a wrong
output rank is ill-typed at the call site rather than a runtime case —
the same bound `leto::stack` demands, which is what lets the host
reference delegate to it from rank-generic code without per-rank shims.
`validate_stack_shape` still checks the rank first for direct callers of
the free function, then runs the fixed order (axis range, non-empty
list, input shape agreement, output shape). No per-input bases are
returned: input `i` always lands at `axis` lane `i`, so backends place
blocks from the enumeration index directly. The output buffer must not
alias any input buffer; both backends reject aliasing with the same
`DispatchFailed` message, and the shared clause pins
rejection-before-mutation. Zero-sized inputs stack vacuously.

`WgpuStackOps` dispatches one block-copy pass per input inside a single
encoder, all passes sharing one pipeline: a thread decodes its input
coordinate, reads through the input strides, and writes the same
coordinate with the new `axis` lane set to the input index. The decode
loop walks padded input lanes while the output has one more lane, so
each input lane maps to its output lane arithmetically (same lane below
the new axis, one higher at or above it); the new-axis lane is seeded
once from the input index, and padding lanes are skipped so no
out-of-range stride lane is ever read. No bounds check is needed because
validation proves each output window equals its input shape exactly. The
same binding-model reasoning as ADR 0073 applies (no portable
multi-buffer binding for a runtime input count), and each pass likewise
holds its pooled uniform buffer until the shared submit. The host
reference delegates to `leto::stack` (ADR 0046) and scatters into the
strided output.

CUDA, ROCm, and Metal do not implement `StackOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`. Those backends
can dispatch single-launch with packed pointer arrays; the WGPU
per-input strategy is a backend-local choice the seam permits, not a
contract other backends must copy.

## Consequences

Coeus can bind `stack_into` on WGPU (and the host reference pair) today.
`HEPH-SHAPE-OPS-PROVIDER-1` narrows further: gather, scatter,
index_select, index_put, masked_fill, tile, where, sort, and nonzero
remain. CUDA/ROCm/Metal implementations and the aggregate fold are
follow-up items.
