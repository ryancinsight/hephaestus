# ADR 0086: Version the existing provider API breaks

- Status: Accepted
- Date: 2026-10-15

## Context

The release gate at merged revision 842a8e96fbfd9bd79b2704374fd4f60878dc79f5
compares 0.20.4 against published 0.20.1. Core and host compare successfully,
but CUDA, Metal, ROCm, and WGPU fail compatibility checks: vector traits now
require a scalar type parameter, the CUDA/ROCm CTC kernel structs were removed,
and WGPU's public QR metadata struct was removed.

## Decision

Version the existing source as 0.21.0, the next breaking pre-1.0 minor,
and keep every workspace package and internal requirement synchronized.
Do not add aliases or restore removed implementation details to conceal the
release boundary. Algorithms, fixtures, tests, and time limits stay unchanged.

Consumers name the vector trait's scalar parameter explicitly and use the
provider's current loss/decomposition entry points instead of removed kernel
metadata. Coeus and Apollo dependency requirements must advance together
before resolving the published provider line.

Verify the full release/API gate, ordinary native tests, and package builds
on the exact final head before publication. The existing Bessel calls also
need Eunomia's compatible 0.9.2 maintenance release; a source-only pass does
not satisfy the registry package gate. Miri remains deferred and unverified.

## Consequences

The manifest now declares the API boundary already present in the source.
This correction adds no further API removal and does not authorize bypassing
reviews or publishing an unverified dependency graph.
