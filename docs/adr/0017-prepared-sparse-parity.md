# ADR 0017 (hephaestus): prepared sparse parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU exposes prepared CSR SpMV and SpMM plans, including multi-RHS SpMV and
one-submit batching. CUDA and ROCm currently rebuild sparse launch metadata and
compile or retrieve kernels on every immediate call, while Metal does not
expose the sparse CSR family even though its device path is WGPU configured for
native Metal.

## Decision

Add backend-owned prepared CSR plans for CUDA and ROCm. Preparation validates
CSR dimensions, dense RHS layouts, output lengths, and fixed buffer bindings;
it retains the backend-native compiled kernel and operation metadata. SpMV and
SpMM share the existing sparse kernels, and multi-RHS SpMV remains an explicit
SpMM alias rather than a second kernel family. A closed enum batch surface
submits mixed prepared SpMV and SpMM plans in order on one native stream.

Add a Metal-owned CSR wrapper and prepared sparse plan wrappers that delegate
to the existing WGPU sparse implementation through the Metal-selected device.
Metal exposes backend-owned buffer and matrix types so WGPU infrastructure does
not cross the public backend boundary.

## Alternatives rejected

- Re-run immediate sparse operations from `dispatch`: this rebuilds prepared
  resources and does not provide repeated-dispatch semantics.
- Upload CSR metadata or dense operands to the host on each dispatch: this
  violates the device-resident sparse contract.
- Duplicate the sparse kernels in Metal: Metal already owns a WGPU-native
  Metal device path, so a second implementation would fork the kernel family.
- Add a generic dynamic backend trait for batches: the operation set is closed,
  so enum dispatch preserves static kernel calls without a hot-path vtable.

## Verification

Backend contracts compare prepared SpMV, SpMM, and multi-RHS SpMV results with
the Leto CPU reference for dense and non-contiguous RHS layouts. Repeated and
mixed batches assert value semantics, and invalid shapes, layouts, output
lengths, and cross-device batches reject before launch. CUDA, ROCm, and Metal
CI will run focused feature, warning-denied Clippy, nextest, doctest, and
rustdoc gates; hardware lanes remain required-device checks when a self-hosted
GPU label is available.
