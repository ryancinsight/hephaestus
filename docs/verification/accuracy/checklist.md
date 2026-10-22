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
- [x] [patch] Preserve earlier WGPU/Metal NaN failures and the f3193138
  terminal results: five workflows succeed (WGPU 116/116); Metal retains
  three NaN violations, with 90 passes, one failure and 21 unrun contracts.
- [x] [patch] Model folded self-equality plus suppressing clamps and preserve
  two rejected comparison prototypes. The private shared-magnitude candidate
  retains finite outputs without changing any fixture or threshold.
- [x] [patch] Independently review the magnitude-selection source delta
  ec650afe and record all six exact-source automatic workflow successes.
  Preserve its compiler limitations and earlier failure history. This does
  not qualify the later private normal-tail delta.
- [ ] [patch] Obtain independent review of the final private diff and assess
  current owner/base overlap plus workflow/deployment side effects before
  any public draft or CI action.
- [ ] [patch] Qualify the restored tail and finite-gradient controls on the
  actual required device backends. The local WGPU Metal attempt already
  returned `AdapterUnavailable` before dispatch: **0/27 lanes executed**.
  Do not retry acquisition through escalation, another adapter or a service.
- [x] [patch] Reproduce the remaining normal-result selection failures on
  the accepted-source baseline; prepare a private scaled-erfc/Gaussian
  formulation and final-reference selector with last-normal controls under
  unchanged tolerances. CPU/source verification does not close device gates.
- [ ] [patch] Independently review and qualify this private normal-tail delta
  on the required backends, including the final two normal-result inputs
  whose reference margins above MIN are smaller than 256 subnormal ULP.
- [x] [patch] Retain independent source/CPU review of provider `838cf7ab` and
  independently accepted observed numerical results for exact5e hosted Metal
  run `39527890349`: full113/219 overlapping suites, six witnesses and complete
  BHHB; preserve all four earlier failures, inputs and -7 controls. Source-
  equivalent package/five Mac/six Linux API results remain scoped to838.
- [ ] [patch] Resolve P1 performance: GELU tail1M pooled H/B1.35333 in that
  complete one-VM cohort. Preserve raw distributions and physical/JIT/portable
  limitations; observed numerical success supplies no performance acceptance.
- [ ] [patch] Independently review new selected-exp sharing providerc0aa and
  its exact validation composition. Raw Naga/MSL exp4→4 and distinct inputs3→2
  support a CSE hypothesis only; renew affected package/API/backend checks and
  matched automatic measurements before accepting this changed source. Keep
  coefficients, selected arithmetic, targets, predicates and budgets unchanged.
- [x] [patch] Prepare private replicated measurement controls: unchanged BHHB
  then fixed HBBH, one future VM/two binaries, all eighteen cases and numeric
  checks, one shared deadline and explicit incomplete-block rejection.
- [ ] [patch] Independently review the replicated helper/spec and external
  analyzer pins, then assess its new automatic cohort with both block ratios,
  all process distributions/drift and no selected pooled speedup claim.
- [ ] [patch] Resolve the six allowed-intrinsic countermodels through an
  owner/coordinator accuracy-contract decision and supporting proof or
  adapter/driver/compiler qualification. Preserve strict existing tests;
  do not increase tolerances to fit observations.
  The private observed-cohort proposal adds exact witness fixtures and retained
  diagnostics only; it does not close this item. Actual six-witness results,
  full-suite results and metadata limits require independent review and an
  owner/coordinator acceptance decision under unchanged targets.
- [ ] [patch] Revalidate all applicable package/API/backend gates and the
  current consumer graph before merge, publication or consumer admission.
- [ ] [patch] Independently review and execute the new-provider validation-only
  Metal comparison after the complete mandatory suite and exact-six
  diagnostics. Bind immutable base/final comparison source, real output
  checks, compiler/adapter identities, raw timing/noise and metadata limits
  on one VM. Preserve existing budgets and targets. An unavailable,
  cancelled, incomplete or unexplained baseline/head result does not close
  numerical or performance qualification.

CPU arithmetic models are not GPU execution receipts. Physical NVIDIA/AMD,
DX12, device performance and portable composition guarantees remain
unverified. Coeus #585, deferred/unverified Coeus Miri, service/process
inspection denials, provider/FFI/Python publication and Harmonia performance
holds retain their coordinator-owned scope.
