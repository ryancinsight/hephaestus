# Architecture decision records

This index is the single lifecycle view for Hephaestus architecture decisions.
Statuses use the canonical `Proposed`, `Accepted`, and `Rejected` terms.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-cuda-backend.md) | CUDA backend composing cuda-oxide and cutile | Accepted |
| [0002](0002-atlas-compute-boundaries.md) | Atlas compute-boundary integration | Accepted |
| [0003](0003-blocked-decomposition-consolidation.md) | Blocked-decomposition host-loop consolidation | Accepted |
| [0004](0004-wg-p4-composite-op-submit-batching.md) | Composite-operation submission batching | Accepted |
| [0005](0005-immutable-wgpu-staging-callbacks.md) | Immutable WGPU staging callbacks | Rejected |
| [0006](0006-wgpu-30-provider-abi.md) | WGPU 30 provider ABI | Accepted |
| [0007](0007-provider-default-msrv.md) | Provider default source and MSRV | Accepted |
| [0008](0008-odd-length-wgpu-storage.md) | Odd-length WGPU storage padding | Accepted |
| [0009](0009-order-preserving-tiled-scan.md) | Order-preserving tiled axis scans | Accepted |
| [0010](0010-eunomia-complex-provider.md) | Eunomia complex provider | Accepted |
| [0011](0011-aequitas-stencil-spacing.md) | Aequitas stencil spacing | Accepted |
| [0012](0012-rocm-backend.md) | Native ROCm backend through HIP | Accepted |
| [0013](0013-backend-volume-parity.md) | Backend-neutral volume ray-integral parity | Accepted |
| [0014](0014-backend-stencil-parity.md) | Backend-neutral 2D Laplacian stencil parity | Accepted |
| [0015](0015-prepared-reduction-parity.md) | Prepared scalar-reduction parity | Accepted |
| [0016](0016-prepared-axis-reduction-parity.md) | Prepared axis-reduction parity | Accepted |
| [0017](0017-prepared-sparse-parity.md) | Prepared sparse parity | Accepted |
| [0018](0018-prepared-map-reduction-parity.md) | Prepared map-reduction parity | Accepted |
| [0019](0019-scan-product-parity.md) | Cumulative-product scan parity | Accepted |
| [0020](0020-metal-random-parity.md) | Metal seeded-random parity | Accepted |
| [0021](0021-metal-fluent-linalg-parity.md) | Metal fluent linear-algebra parity | Accepted |
| [0022](0022-metal-authored-kernel-parity.md) | Metal authored-kernel parity | Accepted |
| [0023](0023-metal-exp-neg-parity.md) | Metal fused negated-exponential parity | Accepted |
| [0024](0024-blocked-pivoted-parity.md) | Blocked pivoted-decomposition parity | Accepted |
| [0025](0025-metal-reduce-into-parity.md) | Metal axis-reduction output parity | Accepted |
| [0026](0026-suffix-scan-parity.md) | Reverse cumulative-sum convenience parity | Accepted |
| [0027](0027-typed-comparison-expression-parity.md) | Typed comparison-expression parity | Accepted |
| [0028](0028-unary-math-expression-parity.md) | Unary math-expression parity | Accepted |
| [0029](0029-error-function-expression-parity.md) | Error-function expression parity | Accepted |
| [0030](0030-exact-gelu-expression-parity.md) | Exact GELU expression parity | Accepted |
| [0031](0031-lgamma-expression-parity.md) | Log-gamma expression parity | Accepted |
| [0032](0032-activation-tail-expression-parity.md) | Activation-tail expression parity | Accepted |
| [0033](0033-dense-vector-backend-parity.md) | Dense-vector backend parity | Accepted |
| [0034](0034-metal-dense-vector-parity.md) | Metal dense-vector parity | Accepted |
| [0035](0035-dense-vector-elementwise-parity.md) | Dense-vector elementwise parity | Accepted |
| [0036](0036-device-local-cow-copy.md) | Accelerator copy-on-write stays device-local | Accepted |
| [0037](0037-uninitialized-device-copy.md) | Overwrite-before-read device allocation | Accepted |
| [0038](0038-blocked-qr-final-panel-synchronization.md) | Finish blocked-QR tail after one readback | Accepted |
| [0039](0039-provider-owned-convolution.md) | Provider-owned accelerator convolution | Accepted |
| [0040](0040-provider-owned-attention.md) | Provider-owned accelerator attention | Accepted |
| [0041](0041-compute-backend-conformance-crate.md) | One generic ComputeBackend conformance crate | Accepted |
| [0042](0042-device-neutral-decomposition-seam.md) | Device-neutral decomposition seam | Accepted |
| [0043](0043-special-value-semantics-capability.md) | Special-value semantics as a dialect capability | Accepted |
| [0044](0044-device-neutral-dense-product-seam.md) | Device-neutral dense product seam | Accepted |
