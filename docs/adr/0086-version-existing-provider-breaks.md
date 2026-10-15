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

## Verified publication

The [guarded merge source](https://github.com/ryancinsight/hephaestus/commit/2f3f6dd04e79f84631acf837b561a24f34e43e2b)
has the identical tree as the validated PR head. All eight actual CI checks
pass, including Metal, WGPU and adapterless CUDA/ROCm. The
[blocking release gate](https://github.com/ryancinsight/hephaestus/actions/runs/38676439939)
and standard verification of all six packages pass; local native verification
is 444 tests with zero skips, plus strict Clippy, docs and doctests.

The [existing trusted publisher](https://github.com/ryancinsight/hephaestus/actions/runs/38677486443)
completed all six 0.21.0 publications. Each registry receipt is not yanked
and identifies that merge source and publisher run:

| Crate | Registry checksum |
| --- | --- |
| hephaestus-core 0.21.0 | `15500707531c360bf8fe9cbf29fbac5b0c73b4301a9936a192897eea8734d146` |
| hephaestus-cuda 0.21.0 | `b5be47157dd95919dd96f2d2868b42dfb0cbe08f7f19d3f914627db7c3e32989` |
| hephaestus-metal 0.21.0 | `5a9e789a136740cf54f48ae4c45c98a0d47558d084a12d75e05b9835983da121` |
| hephaestus-wgpu 0.21.0 | `e9ae690ad223f18d885058223067cb72f3fb3a9f92b15ed22645e2b344894b3a` |
| hephaestus-rocm 0.21.0 | `259eae1f8bc6c316d551ce9e3191d687ca61d5544699d83d9d16e6a72b116ac8` |
| hephaestus-host 0.21.0 | `d20c21cfbf91993ec9b577ce34793358d839fd17e4af789a21b83dde705e829e` |

The provider publication item is complete. Consumer integration remains in
Coeus PR #528 and Apollo PR #669. Their exact-head CI and release gates
remain prerequisites for consumer publication. No consumer release is claimed. Miri remains deferred and unverified; NVIDIA/AMD hardware is also
unverified. Existing test coverage and timeouts are unchanged.
