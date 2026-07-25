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
lengths, and cross-device batches reject before launch. At code head `27bf875`,
CUDA run `30175194827` / job `89722726883` passed in 7m17s, ROCm run
`30175194823` / job `89722726994` passed in 5m44s, and Metal run
`30175194820` / job `89722726870` passed. Hardware lanes were skipped because
hosted GPU labels were unavailable; required-device checks remain enabled for
self-hosted runners. Local Windows package compilation remains blocked before
source compilation by the locked `cutile-rs` refresh and the sibling
Leto/Eunomia `Quantity<T>::in_unit` / `FloatElement` mismatch.
