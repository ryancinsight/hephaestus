# ADR 0004 — WG-P4 composite-op submit batching (norm_l2 / matpow)

- Status: **Accepted (Option B)** — 2026-07-08, user sign-off
- Date: 2026-07-08
- Scope: `hephaestus-wgpu` (`application/linalg.rs`, `application/reduction.rs`, `application/elementwise/`)
- Refs: WG-P4 (backlog, KS-7 perf batch), audit `docs/audit/2026-07-02-hephaestus-gpu-substrate-audit.md` §WG-P4, KS-3/KS-4/KS-4G (backlog, the authored-kernel seam)

## Context

WG-P4 (audit, MED severity): "One encoder + one `queue.submit` per op call (~10-100 µs each);
`norm_l2` costs 3 submits, `matpow` costs *k* submits. ... Fix: encoder-borrowing
`encode_*(&mut CommandEncoder)` layer ... This is also the multi-pass primitive the kernel seam
needs (ADR 0004 [atlas-meta])."

Investigating before implementing surfaced two things the audit's one-line fix suggestion
undersells:

1. **The multi-pass reduction tree is already correctly batched.** `reduction_with_width`
   (`application/reduction.rs:1355`) already creates ONE encoder and threads it through every
   internal tree-reduction pass before a single `submit`. WG-P4's "3 submits" for `norm_l2` is not
   about reduction's own internal passes — it's that `norm_l2` (`application/linalg.rs:1462`)
   composes three separately-submitting *function calls*: `map_reduction` (which itself may be 1-2
   submits internally, first-pass + optional second-pass reduction) and then
   `unary_elementwise_into::<SqrtOp, _>` for the final `sqrt`. Merging these into one submit means
   giving the multi-pass reduction logic an "encode into a caller-supplied encoder" entry point —
   real surgery on `reduction_with_width`'s per-pass buffer-allocation and early-return logic, not
   a thin wrapper.
2. **This project already built the intended fix for this problem class.** `hephaestus-core`'s
   `CommandStream`/`GroupedCommandStream` seam (`domain/stream.rs`, KS-2/KS-4/KS-4G, "done for
   WGPU and CUDA") exists specifically so multiple GPU passes chain into one submit with correct
   inter-pass barrier semantics. But `CommandStream::encode<K: KernelSource<D::Dialect>>`
   (`domain/stream.rs:238`) requires the kernel to implement the newer `KernelSource<Dialect>`
   trait — a different, narrower authored-kernel vocabulary than the `Op: CombineExpr<Wgsl>` /
   `UnaryExpr<Wgsl>` marker-trait dispatch `norm_l2`/`map_reduction`/`reduction`/the elementwise
   family currently use. KS-3 ("Backends consume the core op vocabulary ... Status: in-progress")
   is the tracked, already-owned item for migrating existing ops onto that vocabulary.

So the real fork is: does WG-P4 get a **narrow, ad-hoc fix** (encoder-borrowing variants bolted
onto the existing `reduction.rs`/`elementwise/mod.rs` dispatch functions, independent of the seam),
or does it **ride KS-3's migration** (port `norm_l2`/`matpow`'s constituent ops onto
`KernelSource`, then compose them via `CommandStream`, getting submit-batching "for free" as a
side effect of the already-planned vocabulary migration)?

## Options

### Option A — Ad-hoc encoder-borrowing split (independent of KS-3)

Add `encode_*(&mut wgpu::CommandEncoder, ...)` variants to the elementwise SSOT
(`encode_elementwise` in `application/elementwise/mod.rs` — already a single shared function used
by binary/unary/scalar, so this part is genuinely mechanical) and to `reduction_with_width`
(`application/reduction.rs` — this part is the real work: the function's per-pass buffer
allocation, the `len == 0`/`len == 1` early-return copy paths, and the dynamic
tree-vs-final-pass pipeline selection all need to keep working when the encoder is borrowed
rather than owned). Then rewrite `norm_l2`/`matpow` to open one encoder, call the `encode_*`
variants of their constituent steps, submit once.

- **Pro**: delivers the audit's literal ask now, independent of KS-3's timeline.
- **Con**: builds a second, parallel "encoder-borrowing" convention alongside the seam's
  `CommandStream`, which KS-3 will eventually make redundant for `norm_l2`/`matpow` specifically —
  this is throwaway-adjacent work (not fully throwaway; the elementwise SSOT split is reusable
  regardless of KS-3, but the `reduction_with_width` encode-split is not).
- **Con**: real surgery on `reduction_with_width` (differential-vs-CPU-reference correctness is
  load-bearing here per `numerical_discipline`) for a MED-severity, bounded perf win (audit
  estimates ~10-100 µs per avoided submit — 2 avoided submits per `norm_l2` call).

### Option B — Defer to KS-3, then compose on `CommandStream`

Do nothing on WG-P4 until KS-3 lands `KernelSource` impls for the relevant ops (sum/max/min
combine, sqrt, the map-reduce fused ops). Once those exist, `norm_l2`/`matpow` become straight-line
`CommandStream::encode` sequences with no bespoke encoder-borrowing code to write or maintain —
submit-batching falls out of the vocabulary migration rather than being built in parallel to it.

- **Pro**: no throwaway work; `norm_l2`/`matpow` end up on the same seam as the rest of the ported
  op vocabulary, one dispatch convention instead of two.
- **Con**: blocked on KS-3's completion, which is a separate, larger, already-owned in-progress
  item with its own timeline (owner: claude-seam session, scope `hephaestus-wgpu/**` +
  `hephaestus-cuda/**` per the backlog) — no visibility into how close it is from this
  investigation alone.

### Option C — Narrow ad-hoc fix, scoped to *only* the elementwise SSOT split

Do the `encode_elementwise` split (Option A's mechanical, genuinely-reusable part) now, but leave
`reduction_with_width` untouched and do **not** attempt to fully collapse `norm_l2` to one submit.
This closes zero submits by itself (nothing calls the new `encode_*` variant yet) — it's pure
unused infrastructure until either KS-3 or a future WG-P4 pass gives it a caller, which conflicts
with the project's own "justified constructs" / anti-speculative-generality discipline (a seam
should exist because something needs it now, not preemptively).

- Not recommended for the reason stated: shipping it now would be exactly the "unused
  infrastructure added early" pattern this codebase's own standards flag as slop.

## Recommendation

**Option B.** The perf win here is real but bounded (MED severity, a few hundred microseconds per
`norm_l2`/`matpow` call), and Option A's cost is doing throwaway-adjacent surgery on
correctness-load-bearing multi-pass reduction code in parallel with an already-in-progress
migration (KS-3) that will make that surgery moot for these specific call sites. Recommend closing
WG-P4 as "tracked under KS-3" in the backlog rather than as its own independent item, and
re-opening it only if KS-3 stalls or explicitly excludes the reduction/elementwise op family from
its `KernelSource` migration scope.

## Decision

**Option B accepted (2026-07-08).** WG-P4 is closed as a standalone KS-7 item and re-filed as
tracked-under-KS-3: `norm_l2`/`matpow` submit-batching happens as a side effect of KS-3's
`KernelSource` vocabulary migration once that migration reaches the reduction/elementwise op
family. No ad-hoc `encode_*` surgery on `reduction_with_width` or the elementwise dispatch path
in the meantime. Re-open as an independent WG-P4 item only if KS-3 stalls or its scope excludes
these ops.
