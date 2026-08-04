# ADR 0049: Own accelerator cross-entropy in Hephaestus

- Status: Proposed
- Date: 2026-08-04
- Board item: `HEPH-CROSS-ENTROPY-PROVIDER-1`
- Cross-repository driver: Coeus ADR 0052
- Change class: `[arch] [minor]`

## Context

Coeus currently downloads accelerator logits, evaluates stable row-wise
softmax and mean cross-entropy on the host, stores probabilities in a host
vector, and uploads the backward gradient. Hephaestus owns device buffers,
strided views, validation, compilation, and dispatch but has no classification
loss seam. The selected backend therefore does not own the executed operation.

Leto's CPU contract establishes the mathematical and failure oracle. The
accelerator contract must keep logits, probabilities, loss, upstream gradient,
and logit gradient on the selected provider. Host target indices may cross the
operation boundary once as compact control input, but full tensor payloads
must never transfer or trigger another provider.

## Decision

`hephaestus-core::domain::loss::cross_entropy` owns one device-neutral
forward/backward role over `ComputeDevice`, `StridedView`, real scalar type,
and a validated target-index representation. Forward writes a scalar mean loss
and provider-resident probabilities. Backward additively writes logit gradients
from those probabilities, targets, and a provider-resident upstream scalar.

Shared planning validates rank, nonzero extents, target count and range,
storage spans, writable aliasing, checked products, and backend address width
before compilation or mutation. Prepared provider types retain compiled
kernels and device target storage needed across forward and backward. Backend
ownership is WGPU shader/commands, CUDA source/PTX and launch state, ROCm HIP
source and launch state, and Metal through the existing WGPU-Metal adapter.
Unsupported scalar or device capability returns a typed error and never
downloads, invokes Leto, or selects another provider.

Dispatch occurs once at the operation boundary. Each provider's kernel body is
monomorphic and contains no per-element backend branch or vtable call.

## Alternatives

- Compose consumer-side elementwise and reduction calls: rejected because the
  consumer would retain orchestration, validation, saved-state, and failure
  ownership.
- Implement vendor-specific public loss APIs: rejected because vendor identity
  is a device-implementation dimension, not an algorithm dimension.
- Fall back to CPU for unsupported devices: rejected because it hides a
  provider fault and changes backend identity.

## Verification

One generic conformance suite compares provider forward/backward values to
Leto for contiguous and strided logits, validates target/rank/shape failures,
and checks pre-dispatch failure atomicity. Provider-local source contracts
reject host payload transfers and fallback calls. Warning-denied package gates,
configured Nextest, doctests, SemVer checks, independent review, and exact-head
WGPU/CUDA/ROCm/macOS Metal CI gate consumer cutover.

## References

- [PyTorch cross_entropy contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
