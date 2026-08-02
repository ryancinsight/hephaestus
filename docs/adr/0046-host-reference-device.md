# ADR 0046 — Host reference device and the Leto implementor

- Status: Proposed
- Date: 2026-08-02
- Refs: atlas `backlog.md#atlas-substrate-003` (residual); atlas ADR 0039 §3
  (the Leto–Hephaestus pair's role-trait obligation); ADR 0041 (conformance
  crate); ADR 0042 (the decomposition seam Leto joins first).

## Context

ADR 0039 §3 requires the Leto–Hephaestus decomposition pair to share one
role trait with **Leto and each Hephaestus backend as implementors**. The
role trait now exists (`DecompositionOps`, ADR 0042 stages 1–5) with the
four GPU backends implementing it — but no Leto implementor, because every
seam is parameterized by `D: ComputeDevice` and no host device exists. The
same gap keeps every differential clause in the conformance suite phrased
as "download and compare against a leto call" instead of running one suite
across a CPU reference implementor, and keeps the per-backend
`matches_leto_reference` tests hand-written.

## Decision (recommended)

1. **`HostDevice` in a new `hephaestus-host` crate.** It implements
   `ComputeDevice` with `Buffer<T> = HostBuffer<T>`, a shared handle over
   host memory (`Arc<RwLock<Vec<T>>>`). Interior mutability is required by
   the seam's shared-reference mutation contract (`write_buffer(&self,
   &Buffer, ..)`), and a lock is the honest host analog of a device queue.
   The crate documents itself as the **reference substrate**: correctness
   first, conformance-grade, never a performance path — consumers wanting
   fast CPU execution use leto directly.
2. **Crate placement, not core.** `hephaestus-host` depends on
   hephaestus-core and leto-ops; core stays free of leto-ops and of any
   provider machinery, exactly like the GPU backend crates.
3. **`HostDecompositionOps` first.** The initial implementor surface is
   `DecompositionOps<HostDevice>` adapting leto-ops' fourteen entry points
   — Leto becomes a role-trait implementor per ADR 0039 §3 without leto
   itself learning anything about hephaestus (the adapter lives above it).
4. **The suite instantiates the host pair.** `hephaestus-conformance`'s
   decomposition module runs against `HostDevice` like any backend, giving
   the differential clauses a cross-implementor form: the same clause body
   that checks cuda-vs-leto today checks any-implementor-vs-any oracle,
   and the remaining hand-written `matches_leto_reference` tests migrate
   to instantiations as ADR 0039 §3 requires.
5. **Staged growth.** Other seam families (elementwise, reductions, scans,
   products) gain host implementors only as consumers or clauses need
   them; nothing speculative ships.

## Alternatives

- Host device inside hephaestus-core: rejected — core would grow a
  leto-ops dependency and a second concern (providing) beyond its
  vocabulary role; every GPU backend lives outside core for the same
  reason.
- Implementing the role trait inside leto: rejected — leto sits below
  hephaestus and must not depend on hephaestus-core's device vocabulary;
  the adapter direction (hephaestus-host wraps leto-ops) preserves
  one-way layering.
- `RefCell`/unsynchronized buffers: rejected — `ComputeDevice` requires
  the buffers to be usable where the seams put them (including Send/Sync
  contexts in batch and prepared paths); `RwLock` gives that without
  unsafe.

## Consequences

Every conformance module gains an optional CPU reference instantiation
path; SUBSTRATE-003's residual reduces to the adapter plus the
differential-migration sweep; and future substrate pairs (ADR 0039's
other obligations) have the pattern: role trait in core, reference
implementor in hephaestus-host, GPU implementors per vendor crate.
