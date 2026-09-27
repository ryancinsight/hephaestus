# ADR 0065: Device-neutral embedding-table gather seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-embedding-gather-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

The consumer audit named embedding-table gather as a gap: no accelerator
path exists, so coeus downloads full operands to host for the lookup layer.
`table` is a flat `[num_embeddings, embedding_dim]` row-major buffer;
`indices` selects rows into a contiguous `[n, embedding_dim]` output.

The indices are themselves device-resident data (typically the output of a
prior kernel, e.g. tokenization), so — unlike a host-known shape mismatch —
an out-of-range index cannot be validated before dispatch. Silently
clamping or wrapping it would be defect-masking (error-handling restraint);
the seam must detect and report it.

## Decision

`EmbeddingOps<D, T>` in `hephaestus-core::domain::embedding` provides
`gather_into`. `HostEmbeddingOps` rejects an out-of-range index immediately
(direct memory access, no dispatch step to complete first).
`WgpuEmbeddingOps` cannot fail mid-kernel the same way: it detects the
condition per-thread and records it in a one-`u32` atomic error flag
(`atomicStore` on first detection; the offending thread's own gather is
skipped), downloaded after the dispatch completes. A nonzero flag becomes
the same typed `InvalidConfiguration` error the host path returns. This
costs one 4-byte allocation and one 4-byte download per call — negligible
next to the gather itself — and is a real safety mechanism, not a
placeholder: every out-of-range access is caught, not merely "usually"
caught, because every thread that would read one participates in the flag.

`WgpuGatherMeta`'s uniform is a single `vec4<u32>` even though only one
value differs per call — following ADR 0063's alignment lesson from the
first line rather than rediscovering it a third time.

CUDA, ROCm, and Metal do not implement `EmbeddingOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `gather_into` on WGPU (and the host reference pair) today.
CUDA/ROCm/Metal implementations, the aggregate fold, and the gradient
counterpart (scatter-add for the embedding backward pass, named in the
original backlog item but not delivered here) are follow-up items.
