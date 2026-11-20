# ADR 0089: Version the Leto 0.51 accelerator boundary

- Status: Accepted
- Date: 2026-11-19
- Change class: [major]

## Context

The Hephaestus 0.28.0 archives accept older Leto versions and cap Eunomia
below 0.12. Hephaestus exposes Leto array layouts and Eunomia scalar
bounds in public operations. Core also accepts Aequitas quantities.
Equivalent field layouts do not make values from different releases of
these provider crates interchangeable in Rust. The source delta since
the 0.28.0 archives also removes the ROCm `axis_reduction` module path.

## Decision

Release Core, Host, WGPU, CUDA, ROCm, and Metal together at 0.29.0.
Move direct consumers to Leto and Leto Ops 0.51.0, Eunomia 0.12.1,
and Aequitas 0.4.0. Import ROCm axis reduction functions from the crate
root. The Python package keeps its version and remains wheel-only.

Maintaining the old dependency bounds would allow two incompatible sets
of types. A conversion layer would add copies or duplicate public types
without fixing provider identity. The versioned dependency boundary
keeps the existing numerical contracts and storage layouts.

## Verification

The release gate compares the six public packages with 0.28.0. Package
the exact head, run the existing host, Metal, software WGPU, CUDA and
ROCm adapterless checks, and publish Core before its consumers. WGPU
precedes Metal. Exercise the published Core and Host together with Leto
Ops on a typed finite-difference stencil. Report hardware coverage
from the actual CI jobs.

## Aequitas 0.5 and Leto 0.52

Hephaestus 0.30.0 advances exchanged physical quantities to Aequitas
0.5.0 and array layouts to Leto 0.52.0 with Eunomia 0.12.1. Select
these versions together when moving values between a backend and
its caller. The six backend and core archives match merge commit
`2cb352c52b0a908ef343dd3844248a372ac6506d` and their crates.io
checksums. The release API check compares against 0.29.0. Native
Metal, software WGPU, and adapterless CUDA and ROCm checks pass.
[Release workflow](https://github.com/ryancinsight/hephaestus/actions/runs/44781764295).
