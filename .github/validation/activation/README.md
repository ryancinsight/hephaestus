# Bounded activation evidence

This validation-only helper compares actual baseline
`6f4dd9b942d083a8183fc56c3502c8d8322277d1` with immutable product head
`838cf7abb156b21b521cdc9ef52b14833305cd17`, tree
`26e0a45c44f638ccdb02646297ac71dc9af8e623`. This proposed provider is an
ancestor of the validation revision so the normal automatic route can retrieve
its exact object; the public #509 head remains `a735c608` during preparation.
The validation PR checkout, including GitHub's possible merge commit, is
recorded separately. The spec pins both trees, all eight exported operators
in three dialects, and the unchanged 228-package external lock graph. Relative
to the base, only WGSL GELU, GELU-gradient and tanh-GELU-gradient expressions
differ. Relative to former head `9f7be898`, only GELU's existing factored
branch selector changes from `x < -13` to `x < -7`; the gradient stays at -13.

The previous hosted comparison retains four GELU far-band failures shared by
base `6f4dd9b9` and head `9f7be898`. The new source proposal preserves those
failures and adds all four inputs and -7 handoff neighbors to conformance.
It changes no benchmark, baseline predicate, numerical target or budget.
New full-suite, exact-six and matched comparison evidence is required; CPU
source models and preparation supply no new device or performance result.

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

## Measurements and numerical rejection

The same production `WgpuDevice::try_metal` route runs serial base/head/head/base.
The first unavailable adapter, runtime fault, identity mismatch or unexpected
computed-value failure stops the campaign. A successful acquisition must report
Metal plus IntegratedGpu or DiscreteGpu. Adapter, enabled features and device
limits must match the focused witnesses and all four invocations. The collector
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
retained. No automatic performance threshold or acceptance claim is introduced.

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
python3 -B .github/validation/activation/test_validation.py
```

The verifier rechecks file hashes, binary/source/fixture bindings, exact expression
coverage, all checks/samples, baseline classifications and cohort identity. A
complete measured packet remains pending independent performance assessment.
Unavailable or incomplete evidence leaves performance unmet. Portable intrinsic,
broader provider/consumer, Miri, service, publication and merge holds remain
separate; this helper does not resolve them.
