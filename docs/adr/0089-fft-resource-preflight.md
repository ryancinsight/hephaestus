# ADR 0089: Allocation-free FFT capacity preflight

- Status: Proposed
- Date: 2026-11-18

## Context

A dense split-complex FFT can need more storage than its input volume. For an
axis of length N with B independent lines, Bluestein needs M × B workspace
elements per component, where M is the least power of two at least 2N − 1.
Radix stages share the maximum staging size across the active axes. Existing
preparation only checks these requirements after the consumer allocates its
operands. Consumers need the same check before allocating them.

## Decision

Expose shape-only planning and a WGPU requirements query, with the scalar
bound providing the exact element width. One geometry and sizing path selects
strategies, checks buffer and shader limits, and supplies preparation with its
workspace size. Each reported complex component has its own storage buffer.
Required bytes equal the element count times the scalar width. Dispatch
geometry derives from ceiling division by the kernel's workgroup size, or
the existing two-dimensional split of the batch for fused transforms.

Return a typed capacity violation from both entry points. Its required and
limit fields use the units of the named limit. A dedicated variant changes the
exhaustive public error match contract. Representability, alias, capability,
and allocation errors remain separate events.

## Alternatives

Tracking requirements in each consumer duplicates the FFT strategy and cannot
cover shape-dependent provider changes. Returning a human-readable message
makes consumers parse formatting and loses the limiting quantity.

## Verification

Check exact and adjacent buffer and dispatch boundaries against hand-derived
counts. Recheck device preparation and shape-only preflight with the same
physical device limits. Run the public API compatibility comparison before
selecting the release version.
