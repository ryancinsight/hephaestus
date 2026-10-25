# Bounded activation evidence

This validation-only helper compares immutable base
`6f4dd9b942d083a8183fc56c3502c8d8322277d1` with immutable product head
`c0aa1bdc9a91f876a198700235ed2b2e6197d0db`, tree
`6c8ab23559daffa9a74a9b53f85b8a6cca128bfd`. The provider commit is an
explicit ancestor of this validation revision so the ordinary automatic route
can retrieve its exact object. Public production #509 remains `a735c608`.
The validation PR checkout, including GitHub's possible test merge, is recorded
separately. Exactly three head fields change in the comparison spec: revision,
tree and the actual retained 24-row expression-export digest. Immutable base,
228-row external graph, 372 baseline predicates, fixtures, scorer, timing,
statistics, workflows and budgets remain unchanged.

The previous exact #510 head `5e2d7fd7` compares provider `838cf7ab` in hosted
Metal run `39527890349`. Its full mandatory suites, six witnesses and both
candidate invocations pass the existing numerical checks. The four previous
normal GELU misses from run `39520852311` remain in its measured inputs and
pass. Baseline failures and all earlier incomplete packets remain preserved.
Observed numerical acceptance is scoped to that recorded cohort: the Apple
Paravirtual Metal adapter exposes neither physical backing nor native JIT.
Performance remains P1-held: GELU tail at 1,048,576 elements has pooled batch
medians 2.175250 ms base and 2.943833 ms head, ratio 1.3533308815. Four processes
on one VM do not establish causal or universal effects. Source-equivalent
provider838 package and five Mac/six Linux registry API evidence remain valid
for that source; none transfers to this changed provider.

The new selected-exp sharing proposal preserves GELU's existing x < -7
selector, gradient's x < -13 handoff, coefficients and selected arithmetic
within the explicit strict CPU models. Relative to838, only the WGSL GeluOp
export changes; the other 23 operator/dialect exports remain byte-identical.
Pure Naga30.0.1/MSL still emits four exp nodes/calls before and after; distinct
exp inputs decrease from three to two. This exposes a downstream CSE hypothesis,
not measured instruction elimination, native-JIT identity or a speedup.
Fresh independent source review, affected package/API checks, full backend
suites and matched automatic measurements remain required. No numerical target,
fixture, tolerance, baseline exception or workload changes for this proposal.
Portable, physical, consumer/FFI/Python/service and release holds remain separate;
Coeus Miri remains DEFERRED_UNVERIFIED. This preparation acquires no device and
performs no public push, dispatch, merge or publication.

The existing Metal job appends this phase after all mandatory suites and the six
focused intrinsic witnesses succeed. Both products build on that VM with installed
Rust 1.95.0, the existing release profile, the same feature graph and at most two
Cargo jobs. Exact objects missing from the shallow checkout are fetched by fixed
SHA into disposable clones only. Cargo resolves offline with unchanged locks.
The helper/exporter live outside Cargo target discovery; only disposable clones
receive the explicit benchmark target. Distinct binaries are copied and hashed
before another build can replace the shared target output. Actual compiled
expression exports and CPU reference manifests are checked before any acquisition.

## Invocation and clocks

The reviewed automatic PR step calls:

```sh
python3 .github/validation/activation/run.py --hosted \
  --source "$PWD" --work-root "$RUNNER_TEMP/hephaestus-activation-work-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" \
  --artifacts "$HEPHAESTUS_METAL_VALIDATION_ARTIFACTS"
```

`GITHUB_ACTIONS=true`, `GITHUB_EVENT_NAME=pull_request`, macOS, the canonical
repository, a run ID and `HEPHAESTUS_HOSTED_ACTIVATION_INTENT=reviewed-metal-validation`
are required. Local preparation uses `--prepare-only` instead, plus an explicitly
private `CARGO_HOME`; it runs only the binary's default CPU manifest mode. It never
enters the device campaign. Scratch and artifacts must be outside the source tree,
new scratch/activation directories are exclusive, and a failed attempt is not replayed.

The first user step writes `job-budget.json` and exports
`HEPHAESTUS_METAL_JOB_STARTED_EPOCH`. This clock begins after runner setup, rather
than claiming GitHub's exact job-start time. With the unchanged 2,700-second job,
the phase requires at least 1,500 raw seconds remaining: 1,320 for the helper,
120 for artifact upload and a conservative 60-second setup margin. Insufficient
time fails as unmeasured. Each build is capped at 300 seconds; each invocation at
120 seconds, with 15-second device-completion waits inside the harness. The phase
deadline also bounds every subprocess. A timeout kills the process group, retains
partial logs and stops without another acquisition. No timeout or budget is raised.
Before the reversed block starts, at least four invocation caps (480 seconds)
must remain on the original phase deadline. This admission does not guarantee
verification/finalization fits: two maximum builds plus eight maximum children
require 1,560 seconds before overhead, exceeding the unchanged 1,320-second phase.
Insufficient time starts no fifth acquisition and leaves replication unmeasured.

## Measurements and numerical rejection

The same production `WgpuDevice::try_metal` route runs mandatory base/head/head/base
then reversed head/base/base/head in one future automatic job allocation. The two
optimized binaries are built once there and their preserved hashes serve all eight
serial processes; no prior VM or binary is resumed.
The first unavailable adapter, runtime fault, identity mismatch or unexpected
computed-value failure stops the campaign. A successful acquisition must report
Metal plus IntegratedGpu or DiscreteGpu. Adapter, enabled features and device
limits must match the focused witnesses and all eight invocations. The collector
records Rust/Naga, host, SDK and any standalone Metal tool metadata; unavailable
runtime shader compiler/driver versions remain unknown. A missing standalone
Metal tool does not trigger installation or substitute for runtime identity.

Each of the three operations covers ordinary, negative-tail and finite-saturation
inputs at 65,536 and 1,048,576 elements. The 4,096-point deterministic seeds plus
directed edges and every repeated output are checked before and after timing.
Existing input-dependent 16/32/128/256-ULP and root absolute bounds remain unchanged;
shared root endpoints meet both existing criteria. Normal/zero reference selection
retains the existing subnormal-result exclusion and records those observations.
Two warmups precede nine samples of eight dispatches per case. Timing includes
public dispatch and device completion, excluding compilation/upload/readback.
Raw samples, medians, range, median absolute deviation and head/base ratios are
retained separately for both blocks. The original BHHB summary stays unchanged;
the HBBH block adds H7/H4 head drift and B6/B5 baseline variation. All eight raw
process arrays, monotonic phase offsets, UTC bookkeeping and child elapsed/RSS
remain available. A secondary all-eight pooled ratio never replaces either block
or its drift. Four processes per cohort and nine dependent samples per case share
one VM and persistent caches; they are not independent VM/device replicates.
No automatic performance threshold or acceptance claim is introduced.

Expected baseline failures are a fixed, reviewed list of exact operator, band,
input-bit and failure-class predicates. `predict_baseline.py` derives them from
actual baseline exports and compiled f64 reference bits using explicit binary32
CPU operations. The original direct-libm-exp profiles remain unchanged. A general
fixed alternative uses `RN32(exp2(RN32(a * RN32(log2(e)))))`, with independently
derived binary32 `log2(e)` bits `0x3fb8aa3b`; both intrinsic profiles have gradual
and per-operation FTZ modes. No observed output or input-specific model branch is
used. The generated union preserves all original per-input/class predicates.

Before regeneration succeeds, every active finite exp argument in the fixed
compiled fixture set passes an 80-digit, outward-bounded check against WGSL's
`(3 + 2*abs(argument))` ULP allowance for exp and exp2. Exact-result overflow and
underflow use explicit analytic bounds. Selected values and conditions determine
active dependencies; unselected exp calls are not mistaken for active evidence.
The check validates the unflushed approximation first and accounts separately for
WGSL-permitted subnormal flushing. The smallest normal remains normal; only
subnormal results are flushed by the unchanged FTZ mode. Nonfinite arguments are
reported separately from finite accuracy proofs. Repeated arguments are cached;
proof counts and a canonical record digest bind regeneration to this fixed set.
The optional `--admissibility-output` retains full CPU proof records.

The source profile demonstrates allowable baseline failures, not an actual JIT
mechanism or whole-domain numerical guarantee. The host libm exp/exp2 models are
neither correctly-rounded-f32 nor portable/compiler qualification. The generated
map is checked before the hosted campaign; it is never learned from new device
results. Baseline exit 1 is accepted only when all failures match those predicates
and counts reconcile. Any unlisted baseline failure and every head failure stop
the campaign. Predicate matches preserve failures as evidence and confer no
baseline correctness acceptance.

## Resources and evidence

Machine records use explicit line-buffered I/O without retaining stream locks
across device work. Write or flush failures exit 3, distinct from numerical
failure 1 and acquisition failure 2; they cannot qualify as expected baseline
failures.

One case owns buffers at a time; the fixed maximum has at most 16 MiB logical
host/input/output/readback f32 storage, plus under 1 MiB seed/reference/failure
summaries. Driver/compiler/staging caches are additional. Child completion
`wait4` records peak RSS (Darwin bytes; Linux KiB converted to bytes), user CPU
and system CPU usage. This is observational accounting, not hard OS memory
enforcement or a whole-host/process-tree memory cap.

During local CPU preparation, native `/usr/bin/time` reported EPERM reading
`kern.clockrate` and returned exit 1; retained stdout proves the Git child emitted
its head hash. The metric-query ordering is not established. The optional
`sysctl -n hw.memsize` read also failed with EPERM and was incorrectly repeated
once in the next preflight. Both failures are retained externally. All such
queries are removed; physical host RAM remains null/unavailable, with no retry
or recovery through another API. `wait4` uses ordinary child completion data and
does not query those denied host metrics. No local device acquisition occurs.

The narrow artifact includes source/spec/predicates, locks, distinct binaries,
actual exports, bounded manifests, every subprocess stdout/stderr/resource record,
raw measurements and a SHA-256 file index. It excludes Cargo cache/target trees
and credentials. The existing job retains it on PR success or failure. Verify
downloaded evidence with:

```sh
python3 .github/validation/activation/verify.py --artifacts /path/to/download
python3 -m unittest discover -s .github/validation/activation -p 'test_[rv]*.py'
python3 -B .github/validation/activation/test_adapter_identity.py
```

The unchanged `test_baseline_profiles.py` also requires its four explicit CLI
inputs: actual compiled `--exports`/`--manifest` and accepted
`--previous-predicates`/`--previous-generator` paths. Missing inputs fail rather
than skip that test class; retained exact fixtures may be reused without a build.

The verifier rechecks file hashes, binary/source/fixture bindings, exact expression
coverage, all checks/samples, baseline classifications and cohort identity. The
new complete status requires exactly BHHB+HBBH, global indices 0..7, block/slot
labels, unchanged binary hashes and one matched adapter identity. A failure or
incomplete second block preserves the first as raw diagnostic only; it cannot
produce a complete replicated result or selected favorable summary. Existing
packets retain their original frozen verifier. Complete replicated measurements
remain pending independent performance assessment. Endpoint-role reversal probes
order association; it does not remove arbitrary time/thermal/carryover noise or
prove a CSE mechanism. Equal ordinal means do not cancel pooled-median drift.
Unavailable or incomplete evidence leaves performance unmet. Portable intrinsic,
broader provider/consumer, Miri, service, publication and merge holds remain
separate; this helper does not resolve them.
