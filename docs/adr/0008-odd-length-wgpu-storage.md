# ADR 0008: Odd-length WGPU storage padding

- Status: Proposed
- Date: 2026-07-16
- Change class: patch

## Context

The generic core validators reject a buffer or host slice unless its logical
byte length is a multiple of four. The WGPU backend already allocates storage
at a four-byte-rounded physical size, so this rejection prevents valid typed
`u16` payloads such as the 27-element native-f16 FFT volume from reaching the
provider. The same restriction prevents otherwise valid host upload and full
buffer write operations.

## Decision

Core validates only byte-size overflow. WGPU owns its required physical
alignment: for a logical payload of `n` elements of size `s`, it allocates and
transfers `4 * ceil(ns / 4)` bytes while exposing `len() == n`. It zero-pads
the trailing physical bytes on upload and full-buffer write, and readback
copies only the exact `ns` logical bytes to the caller.

Sub-buffer writes remain bounded by their logical range; an unaligned interior
sub-write must not pad over the next logical element. The immediate consumer
needs full-buffer operations only, but this invariant prevents padding from
mutating adjacent data.

## Invariant and evidence boundary

The padding bytes are outside the typed logical range. Therefore kernels index
only `[0, n)`, and host roundtrips preserve each of the `n` values exactly.
Core unit tests prove the pure size relation; a real WGPU device regression
checks allocation, upload, write, and download for an odd `u16` length. This
is type/contract plus empirical-device evidence, not a proof for every driver.

## Consequences

- Consumers retain exact logical buffer lengths without provider-specific
  padding wrappers.
- WGPU owns its physical alignment requirements behind `WgpuBuffer<T>`.
- Apollo can construct the 3x3x3 native-f16 plan through its direct
  Hephaestus boundary.
