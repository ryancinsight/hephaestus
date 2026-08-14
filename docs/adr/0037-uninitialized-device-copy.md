# ADR 0037: Separate overwrite-before-read device allocation

- Status: Accepted
- Date: 2026-07-28
- Scope: `ComputeDevice` allocation and device-local Coeus COW replacement
- Change class: `[arch]`/`[minor]`

## Context

The device-local COW path allocates a replacement buffer and then copies every
element from the shared source buffer into that replacement. The common
`ComputeDevice` contract previously exposed only zero-initialized allocation.
CUDA and ROCm consequently wrote a full initialization pattern before the
copy, even though no element was read before the copy completed.

## Decision

Add the required `ComputeDevice::alloc_uninitialized_with_hint` operation and a
default-hint convenience method. Its returned buffer is valid only for a
producer that fully overwrites it before any read. Callers requiring defined
contents continue to use `alloc_zeroed_with_hint`.

CUDA allocates with `cuMemAlloc` and ROCm with `hipMalloc`, omitting their
provider memset operations. WGPU uses the same buffer-creation path because
WebGPU controls initialization at the platform boundary; Metal delegates to
its wrapped WGPU device. Unavailable CUDA and ROCm configurations retain typed
errors. The required method makes every shipped backend state its handling
explicitly instead of silently inheriting a zeroing fallback.

## Alternatives rejected

- Reuse zeroed allocation everywhere: rejected because CUDA and ROCm pay a
  full-buffer write before a full-buffer overwrite in the COW path.
- Put raw CUDA/HIP allocation in Coeus: rejected because it duplicates vendor
  policy in a consumer and prevents one shared backend contract.
- Expose an unsafe or untyped buffer escape: rejected because the
  overwrite-before-read rule must remain visible in the typed device seam.

## Verification

Provider contracts allocate through the new seam, fully write a nontrivial
value set, download the result, and assert exact values. Core, WGPU, Metal,
CUDA, and ROCm stub feature-aware compilation and linting are the local gates;
Linux ROCm feature compilation and physical-device execution require hosted
CI. The local CUDA focused nextest build reaches test-binary linking but is
blocked by the checkout's `x86_64-w64-mingw32-gcc` linker; feature-aware CUDA
compilation and hosted CUDA CI remain applicable. The change provides
structural evidence that CUDA and ROCm no longer call memset on this allocation
path. Runtime bandwidth, latency, and resident-memory changes require a
controlled benchmark and are not claimed here. Provider PR #136 exact-head
run `30343728210` passed CUDA job `90224950226`, run `30343728174` passed WGPU
job `90224950173`, run `30343728133` passed ROCm job `90224950310`, and run
`30343728161` passed Metal job `90224950041`. NVIDIA `90224951166` and AMD
`90224950902` skipped because no physical-device runner was dispatched.
Coeus PR #235 merged at `c7fcdc1`; its final docs-head run `30346488092` passed
CUDA `90233799719`, WGPU `90233799768`, ROCm `90233799650`, and Metal
`90233799737`. Required-device ROCm `90233800152` skipped because no hosted
AMD runner was dispatched; physical-device execution is not claimed.

## Revisit trigger

Revisit if a shipped backend cannot provide a real allocation without an
initialization write, or if a controlled COW benchmark shows allocation
initialization is not material relative to the device-local copy.
