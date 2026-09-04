# ADR 0056: Eunomia-owned device layout seam

## Status

Accepted

## Date

2026-09-03

## Board item

`HEPH-EUNOMIA-LAYOUT-SEAM-2026-09-03`

## Context

Hephaestus currently uses `bytemuck::Pod` and `bytemuck::Zeroable` in its
device-neutral contracts, backend implementations, owned ABI metadata, and
byte marshalling. Eunomia now owns the Atlas layout vocabulary and provides
native marker derives plus zero-copy byte views. Keeping both marker families
in Hephaestus makes the public generic contract depend on an incumbent
provider and forces downstream consumers to derive two unrelated layout laws.

## Decision

Hephaestus uses `eunomia::Pod` and `eunomia::Zeroable` for every first-party
device-buffer, kernel-parameter, metadata, and operation-seam contract.
Owned `repr(C)` and `repr(transparent)` values derive the Eunomia markers.
Host/device byte views call `eunomia::layout::{bytes_of, cast_slice,
cast_slice_mut, ...}` directly, preserving the existing zero-copy ownership
and alignment contracts. The core seam and all backend implementations move
together so no local trait, forwarding bound, or conversion layer mirrors the
old API.

The direct `bytemuck` dependency and source imports are removed from
Hephaestus packages by this migration. A vendor crate may still receive
`bytemuck` transitively from an external API such as WGPU; that graph residue
is not a first-party contract and is recorded rather than hidden behind a
second marker or a feature-gated stub.

## Alternatives rejected

- **Keep `bytemuck::Pod` as a public super-bound.** Rejected: it leaves the
  provider-adoption defect in the deepest consumer-facing seam and prevents
  Eunomia-only downstream metadata from satisfying the contract.
- **Add a local `AtlasPod` bridge trait or blanket conversion.** Rejected:
  this mirrors the marker API, creates a second source of truth, and adds a
  compatibility shim instead of completing the co-evolution.
- **Copy values through an intermediate byte vector.** Rejected: it adds
  allocation and memory traffic without changing the layout proof.
- **Delete all transitive `bytemuck` occurrences from external dependency
  graphs.** Rejected: Hephaestus does not own WGPU or vendor crate internals;
  only direct first-party exposure is in scope.

## Safety and performance

Eunomia's unsafe marker contract requires valid all-zero values, no padding or
invalid bit patterns, and `Copy + 'static`; its derive emits field bounds and
compile-time padding checks for concrete owned structs. The existing device
transfer operations remain borrowed byte views, so the migration introduces no
copy, allocation, or runtime dispatch. Any manually implemented marker keeps
its local `// SAFETY:` proof and remains covered by the existing backend and
conformance tests.

## Verification

The migration requires the committed format, locked workspace compilation,
warning-denied Clippy, native Nextest suites for the host and available
backends, doctests, rustdoc, lockfile integrity, and source/manifest scans for
direct Hephaestus-owned `bytemuck` contracts. Public-bound compatibility is a
major change and requires the repository's SemVer gate where its baseline
can be collected. CUDA and ROCm evidence is reported separately when their
toolchains or hardware are available.
