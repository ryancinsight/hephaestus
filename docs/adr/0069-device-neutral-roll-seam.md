# ADR 0069: Device-neutral rank-2 axis roll seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

`HEPH-SHAPE-OPS-PROVIDER-1` bundles fifteen shape/indexing operations (cat,
split, stack, gather, scatter, index_select, index_put, masked_fill, pad,
roll, tile, tril, triu, where, sort, nonzero) as one item. `PadOps` was
already split out and delivered (ADR 0062); this record splits out `roll`
the same way — one self-contained algorithm at a time, each independently
verifiable, rather than one seam attempting all fifteen at once.

## Decision

`RollOps<D, T>` in `hephaestus-core::domain::roll` provides
`roll_axis_into`, matching the family's rank-2/axis convention but with
`output`'s shape equal to `input`'s exactly — a roll never resizes an axis,
unlike `InterpolationOps`/`AdaptivePoolingOps`. The source index for
destination `i` is `(i - shift).rem_euclid(axis_len)`: `rem_euclid` keeps
the result in `[0, axis_len)` for every sign and magnitude of `shift`
(negative, zero, or exceeding `axis_len`), so no special-casing is needed
at either boundary.

`WgpuRollOps` precomputes `shift.rem_euclid(axis_len)` host-side into the
uniform (`shift_rem`, always non-negative and less than `axis_len`) rather
than passing the raw signed `shift` into the shader, so the WGSL body needs
only unsigned wraparound arithmetic (`(dst_idx + axis_len - shift_rem) %
axis_len`), never a signed modulo whose sign convention would have to be
matched to Rust's `rem_euclid` inside the shader. One thread per
`(lane, dst_idx)` pair (flat over `lanes * axis_len`, matching the family's
established dispatch shape) computes its own wrapped source index and
copies independently.

Both implementations validate the complete view before touching storage:
the layout must fit its buffer and the output layout must be injective. The
host implementation rejects aliased input and output buffers before acquiring
either lock, preventing a read/write lock self-deadlock. The WGPU
implementation additionally proves every generated signed shader address is
representable and rejects a dispatch whose workgroup count exceeds the
device's per-dimension limit. The source-index helper computes the shift
remainder without subtracting signed extremes, so `i64::MIN` and `i64::MAX`
remain defined inputs.

CUDA, ROCm, and Metal do not implement `RollOps` yet, and the seam is not
folded into `assert_backend_contract`/`BackendUnderTest` this increment,
for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `roll_axis_into` on WGPU (and the host reference pair)
today. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further: cat, split, stack,
gather, scatter, index_select, index_put, masked_fill, tile, tril, triu,
where, sort, and nonzero remain, each a candidate for the same one-op-at-
a-time split. CUDA/ROCm/Metal implementations and the aggregate fold are
follow-up items.
