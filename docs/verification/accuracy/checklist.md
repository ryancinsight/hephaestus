# Unary accuracy correction checklist

Target version: **0.27.1**. Change class: **[patch]**. Phase: **Execution**.

This checklist tracks the three findings on shipped 0.26.0 tree
`7240c9dc9083b74a6d4d85b1c51840bde18de1b1`. It does not supersede the
historical macOS checklist. Merged 0.27.0 preserves all eight numerical
paths unchanged; this correction targets that current base. The
[analysis](design.md) defines the boundaries.

- [x] [patch] Preserve the independent shipped-source review and the complete
  final-tree sampled WGPU 116/116 and Metal 112/112 plus 217/217 passes.
  The Metal invocations overlap; their counts are not summed. Historical
  Mish failures do not establish a failure of that final source.
- [x] [patch] Reproduce three normal-tail selection failures on the test-only
  baseline; retain erfc 9/9.19 and GELU/gradient -12.6/-13 in the candidate.
- [x] [patch] Bound finite GELU-tanh-gradient operands before eager WGSL
  evaluation; keep both signs, adjacent saturation handoff values, huge
  finite controls, ordinary controls and all original tolerance targets.
- [x] [patch] Combine normal GELU-gradient terms before halving at -13;
  preserve the C/HIP expressions and all public operation/value contracts.
- [x] [patch] Complete scoped CPU/library tests, strict lint/docs/fmt,
  and actual-expression arithmetic checks; retain commands,
  revisions and results in the coordinator's review receipt.
- [x] [patch] Reproduce the post-draft structural-NaN failure with actual
  expression models whose clamp may propagate NaN or select either bound;
  validate the private forwarding correction without removing any fixture.
- [ ] [patch] Review the NaN-forwarding source delta independently before
  pushing it; then qualify the unchanged structural controls on the required
  automatic backend gates. The public draft's earlier WGPU/Metal runs fail.
- [ ] [patch] Obtain independent review of the final private diff and assess
  current owner/base overlap plus workflow/deployment side effects before
  any public draft or CI action.
- [ ] [patch] Qualify the restored tail and finite-gradient controls on the
  actual required device backends. The local WGPU Metal attempt already
  returned `AdapterUnavailable` before dispatch: **0/27 lanes executed**.
  Do not retry acquisition through escalation, another adapter or a service.
- [ ] [patch] Resolve the remaining normal-final/subnormal-erfc-intermediate
  GELU/gradient interval with a scaled formulation or an explicit accepted
  qualification limit; reference-normal filtering is not full tail closure.
- [ ] [patch] Resolve the six allowed-intrinsic countermodels through an
  owner/coordinator accuracy-contract decision and supporting proof or
  adapter/driver/compiler qualification. Preserve strict existing tests;
  do not increase tolerances to fit observations.
- [ ] [patch] Revalidate all applicable package/API/backend gates and the
  current consumer graph before merge, publication or consumer admission.

CPU arithmetic models are not GPU execution receipts. Physical NVIDIA/AMD,
DX12, device performance and portable composition guarantees remain
unverified. Coeus #585, deferred/unverified Coeus Miri, service/process
inspection denials, provider/FFI/Python publication and Harmonia performance
holds retain their coordinator-owned scope.
