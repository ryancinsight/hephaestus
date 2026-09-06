# ADR 0001: Provider-owned CUDA driver boundary

- Status: Accepted
- Date: 2026-06-10; revised 2026-09-05
- Class: [patch] [arch]
- Item: [HEPH-CUDA-DRIVER-BOUNDARY](../../backlog.md#heph-cuda-driver-boundary)
- Parent: [Atlas ADR 0001](../../../../docs/adr/0001-gpu-accelerator-substrate.md)

## Contract and evidence

Hephaestus owns device acquisition, contexts, buffers, transfers and kernel
submission behind `hephaestus-core::ComputeDevice`. CUDA ABI types stay private
to this provider. Mathematical operation families remain generic over the
existing scalar, operation and device seams.

The original dependency prescription did not implement its promised dynamic
loading. At baseline `3aee4fc`, cuda-oxide 0.4.0's build script links the CUDA
import library. NVIDIA CUDA 13.3's Windows library requests LIBCMT, conflicting
with the Rust target's dynamic CRT. Apollo's all-feature dependency audit also
rejects cuda-oxide's GPL-3.0-or-later license under its existing allowlist.

There is an independent safety failure: cuda-oxide's generated `size_t` is
`c_ulong`, 32 bits on Windows x64. `current_memory_info` passes two such locals
to `cuMemGetInfo_v2`, whose CUDA header signature writes `size_t` outputs,
64 bits on that target. The same width mismatch invalidates the dependency's
`CUDA_MEMCPY2D` layout. The installed CUDA 13.3 `include/cuda.h` (the
`cuMemGetInfo_v2` version mapping and `cuMemGetInfo` declaration) supplies the
ABI reference. A compile baseline passes; executing this known mismatch is
not a valid pre-change test.

## Decision

Remove cuda-oxide and define the required CUDA driver boundary in
`infrastructure/driver/`, partitioned by context/device, memory and kernel
operations. Declarations follow NVIDIA's header: C integer widths,
pointer-sized byte counts and output storage, opaque handle pointers, and the
platform CUDA calling convention. Compile-time assertions pin the target
layout obligations. This is a native replacement; no mirrored `sys` adapter
or forwarding compatibility surface remains.

The provider supports the 64-bit CUDA ABI. Windows loads the driver from
System32; a missing requested file is absence, while dependency and loader
faults remain errors. Linux classifies only the requested soname's explicit
ENOENT diagnostic as absence; unfamiliar diagnostics remain faults.

A process-owned driver table resolves symbols once and retains its loaded
library. Every acquired context keeps a reference to that table; buffers and
modules preserve the context's ownership. Function pointers cannot outlive the
library. Calls select the already-resolved operation at their existing device
boundary, with no lookup inside mathematical loops.

Driver absence, missing required symbols, initialization errors and operation
errors remain distinguishable. Errors preserve the operation or symbol name
and numeric driver status. An installed but broken driver is a fault, not a
reason to fabricate an unavailable adapter or quietly select another backend.
Drop-time failures emit structured operation/status events through the existing
locked tracing package. Destructors retain resources whose release cannot be
established safely and preserve the calling thread's current context.

Rust compilation requires neither CUDA headers nor an import library. Device
execution requires an installed NVIDIA driver. Existing runtime-generated
CUDA kernels additionally need NVRTC; that compiler's lifetime is a separate
owned resource. Kernel authoring and numerical algorithms do not change in
this increment.

## Alternatives

Retaining cuda-oxide and widening only the memory-info locals leaves other ABI
and static-linkage defects in place. A local dependency patch or a mirrored
API merely moves that unsupported binding downstream. Replacing it with
another third-party device abstraction duplicates the role Atlas assigns to
Hephaestus. The existing libloading dependency is sufficient for the native
boundary; no new provider or service is required.

## Verification and rejection criteria

- Compile-time ABI checks and independent review compare every imported
  signature with NVIDIA's declarations, including output widths and lifetime.
- Physical-device contracts cover memory capacity/free bytes, zero-length and
  strided transfer, context restoration, pinned asynchronous transfer, kernel
  launch, and concurrent acquisition. Existing numerical oracles remain.
- Real missing-library and missing-symbol calls distinguish loader errors;
  tests do not substitute fake driver computation.
- Warning-denied checks, bounded Nextest, docs and consumer integration run
  against the final revision. Apollo's activated all-feature graph contains
  no cuda-oxide and passes its unchanged license policy.
- Reject changed public semantics, swallowed driver faults, a dangling library
  handle, static CUDA linkage, warning suppression, or narrowed workloads.

Miri cannot execute driver FFI; native GPU tests and header-grounded ABI review
provide complementary evidence, not a formal proof of driver correctness.

The required-device Windows run `dde33b92-6f4b-4f3d-b41f-d6ec22cfab62`
passes 305 CUDA/host contracts with zero skips on an RTX 5080, driver 610.47.
The first run identified two source-location guards needing migration and an
incorrect null-context test precondition. CUDA 13.3 `cuda.h:6603-6612` defines
`cuCtxSetCurrent(NULL)` as one stack pop. The regression now asserts an empty
stack on a fresh thread before and after final-resource destruction; the
independent live-context preservation assertion remains unchanged.
The local evidence directory is `output/cuda-driver-boundary` under Atlas's
bounded retention policy; source hashes identify the tested implementation.

The comparative smoke exposed an independent reference defect. For its
unchanged `2^20` samples repeating the integers `-8..=8`, the exact squared
sum is `61_680 * 408 + 344 = 25_165_784`. Each term and that result are
representable in binary32. Serial binary32 accumulation instead produces
`25_001_304`; its rounded norm differs from the analytical norm by about
0.33%, exceeding the existing `1e-5` relative assertion. The benchmark oracle
therefore derives the repeating-period sum; the input, timed operations and
tolerance remain unchanged. A full-length contract checks both immediate and
prepared norms against that independent analytical reference.
That contract passes debug run `8aa84e12` and release run `da7a3e40` with
exact equality. The full comparative smoke executes all operations in 0.6
seconds under its unchanged 60-second supervisor; this is execution evidence,
not a controlled CUDA performance comparison.

## Revision

2026-09-05: replace the contradicted dependency prescription with provider-owned
ABI and runtime loading. This corrects the Windows output-size defect and
restores the original no-toolkit compilation contract. Git preserves the
superseded dependency decision; the current record governs this replacement.
