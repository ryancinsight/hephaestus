# ADR 0053: Own multidimensional accelerator FFTs in Hephaestus

- Status: Accepted
- Date: 2026-08-26
- Revision 2026-08-28: Consumer-domain grouped kernels with fixed buffers,
  parameters, and geometry can retain `WgpuBoundGroupedDispatch` values beside
  a prepared FFT. Binding validates the grouped interface and fixed-resource
  device provenance; every encode validates the target sequence's device
  provenance before reusing the pipeline, parameter uniform, bind groups, and
  dispatch grid. This
  lets Apollo retain STFT framing/window/overlap kernels around provider-owned
  FFT commands without rebuilding Hephaestus binding state on the warm path.
- Revision 2026-08-28: Apollo STFT established a batched selected-axis
  requirement: dense C-order `[frame_count, frame_len]` storage must transform
  axis 1 without transforming across frames. Prepared FFT plans therefore
  validate and retain an explicit nonempty set of unique in-range axes.
  Existing preparation remains the all-axis convenience over this canonical
  seam. Inverse normalization uses the product of active extents, not the full
  operand element count. Empty, duplicate, and out-of-range selections fail
  before provider allocation or operand mutation.
- Revision 2026-08-28: The WGPU provider now instantiates one sealed,
  scalar-generic FFT plan for `f32` and native `eunomia::F16`. Binary16 requires
  `ShaderF16`; preparation rejects the missing capability before plan
  allocation or operand mutation and never widens or falls back to the host.
  One WGSL template serves both representations, with scalar-generic complex
  butterflies expressed as two-lane vectors. Root and reciprocal tables are
  evaluated once during preparation and narrowed directly to the selected
  storage scalar; no transform shader performs wider arithmetic. The compact
  staged table stores one half-circle and reconstructs radix-four's third root
  by sign reflection. On an RTX 5080 through Vulkan, paired forward/inverse
  Criterion regression-slope time estimates (95% confidence intervals) are
  162.71 microseconds [161.25, 164.67] for binary32 and 171.31
  [168.20, 174.41] for binary16 at 65,536 elements; for 64-cubed they are
  221.64 [219.22, 225.42] and 215.52 [212.69, 218.68] microseconds
  respectively. Binary16 is higher in the long staged regime and lower in the
  fused workgroup-local regime; neither result
  establishes a universal scalar-width speed claim.
- Revision 2026-08-26: The executable WGPU increment prebinds pipelines,
  immutable parameter buffers, bind groups, and dispatch grids during plan
  creation. The provider-neutral `FftOps::encode_fft` records into a
  caller-owned command stream, so composed consumers allocate no FFT-owned
  transient resources and can submit the transform with adjacent work.
  Bluestein phases are reduced modulo `2N` in integer arithmetic, evaluated in
  `f64`, narrowed once, and uploaded as direct complex factors. Benchmark
  submission and readback use explicit deadlines, and the required-device CI
  job executes the benchmark binary under a 60-second process bound.
- Revision 2026-08-26: Prepared WGPU plans own cloned handles to their fixed
  operands and bind pack/unpack commands directly to those allocations. This
  removes two full-volume scratch allocations (`8N` bytes for split `f32`) and
  four full-volume device copies (`16N` bytes per warm transform). The concrete
  WGPU plan also encodes into a provenance-carrying consumer sequence so
  Kwavers PSTD can retain one pass across adjacent kernels and FFT stages
  without accepting a separately asserted device identity.
- Revision 2026-08-26: The matched 256x128x128 lossless PSTD workload exposed
  the staged radix plan as structurally unsuitable for bounded power-of-two
  axes: six forward/inverse pairs took 13.810 ms median, exceeding Kwavers's
  complete 10.09 ms step before any physics kernels ran. WGPU plans now select
  a device-limit-checked fused workgroup radix for power-of-two axes up to 1,024
  elements. Each active axis becomes one dispatch with no pack/unpack passes or
  full-volume workspace; singleton axes emit no command. The same benchmark is
  7.9974 ms median (42.1% lower), and a forward/inverse plan pair replaces 64
  MiB of full-volume workspace at this shape with two 4 KiB root tables. The
  staged radix and Bluestein strategies remain for unsupported lengths or
  limits.
- Board item: `HEPH-FFT-PROVIDER-1`
- Cross-repository drivers: Apollo GPU FFT and Kwavers PSTD/backend selection
- Change class: `[arch] [minor]` in Hephaestus; breaking consumer cleanup in
  Apollo and Kwavers

## Context

The CPU and accelerator FFT paths have different current owners. Apollo owns
the canonical CPU Fourier algorithms over Leto arrays, but it also owns a WGPU
three-dimensional split-complex plan. Kwavers re-exports that plan while also
owning a second WGPU FFT inside its PSTD solver. The Apollo plan supports
power-of-two and Bluestein axes but advertises no one- or two-dimensional WGPU
plan; the Kwavers kernel supports only bounded power-of-two axes. This leaves
two accelerator implementations, incomplete dimensional contracts, duplicated
twiddle/workspace state, and no truthful `Leto`/`Hephaestus` selection boundary.

Leto's accepted architecture makes it the CPU array, storage, and layout
substrate. Apollo owns Fourier arithmetic on that substrate. Hephaestus owns
device buffers, command submission, prepared kernels, and accelerator operation
families. Moving FFT arithmetic into Leto would invert the existing
Leto-to-Apollo dependency direction; retaining accelerator kernels in either
consumer would preserve the duplication.

## Decision

`hephaestus-core::domain::fft` owns one device-neutral dense complex FFT
contract over `D: KernelDevice`, scalar storage, and const rank. The admitted
ranks are one through three. A plan records a nonzero checked shape and the
product of its axes. Operands are two same-layout `StridedView`s for real and
imaginary components. The first provider contract admits dense C-order storage;
unsupported striding returns a typed validation error before preparation or
mutation. Components cannot alias. Empty axes, product overflow, address-width
overflow, length mismatch, and unsupported rank or scalar are typed failures.

The numerical convention is fixed rather than configurable: the forward
transform uses the negative exponential and no scale; the inverse uses the
positive exponential and scales by the reciprocal of the full transformed
element count. This is Apollo's FFTW-compatible convention and makes
`inverse(forward(x)) = x` within the scalar's derived floating-point bound.
Direction is selected when preparing an operation, outside the dispatch loop.

`FftOps<D, T>` exposes an associated prepared type parameterized by rank.
Preparation validates operands, selects every axis strategy, compiles pipelines,
and allocates provider-owned scratch. Dispatch only encodes/submits work against
caller-owned device buffers; repeated dispatch performs no allocation, pipeline
compilation, capability probe, host transfer, or provider switch for FFT-owned
resources. A standalone dispatch still creates the command encoder required for
one WGPU submission; composed consumers call `FftOps::encode_fft` with an
existing provider command stream. WGPU consumers already composing raw WGSL
kernels may use the concrete grouped sequence's raw pass between prepared-plan
encodes; the sequence carries its owning device so the FFT validates actual pass
provenance. The operation still owns its validated operand handles and all
prebound commands. The prepared operation remains bound to the validated
operand storage and shape, matching the existing Hephaestus prepared-operation
model. A consumer that must upload input before creating either stream calls
`WgpuPreparedFft::validate_device` first. This exposes the same provider-owned
identity check and typed foreign-device rejection used by dispatch and encoding;
it does not replace their per-operation provenance validation.

`hephaestus-wgpu` is the first executable provider. It consolidates Apollo's
general radix/Bluestein implementation and Kwavers's PSTD requirements into one
rank-generic axis-pass plan. Power-of-two axes up to 1,024 elements use one
workgroup-local dispatch when the acquired device exposes the required 64
invocations, 12 KiB workgroup storage, and dispatch grid. Other power-of-two
axes use the staged global-memory radix plan; arbitrary axes use Bluestein.
Singleton axes are identity operations and emit no command or workspace.
One-dimensional and two-dimensional transforms are
not wrappers around a three-dimensional public API: all ranks instantiate the
same axis planner, and only their actual axes are dispatched. All axis passes
are encoded into one command stream submission. Bluestein coefficient
preparation is Hephaestus-owned; integer modulo reduction bounds the quadratic
phase before `f64` evaluation and one narrowing to the selected storage scalar.
The WGPU implementation admits `f32` and native `eunomia::F16` through the
sealed `WgpuFftScalar` contract. Binary16 requires the acquired device to expose
`ShaderF16`, and capability rejection precedes operand validation, allocation,
pipeline compilation, or mutation. The binary16 WGSL token remains local to
this sealed contract rather than becoming a global `DialectScalar<Wgsl>`
implementation: other generic WGPU operation families do not yet enable or
validate native half execution, so admitting the type globally would expose
unsupported shader paths. It cannot call Apollo or silently execute
the requested transform on the host. Metal continues to use
the WGPU provider over its selected adapter. CUDA and ROCm implement the same
seam in later provider increments; unsupported providers return a typed
capability error.

Kwavers owns the closed runtime selector `Leto | Hephaestus` at its Fourier
operation boundary. `Leto` names the host array route and delegates Fourier
arithmetic to Apollo over Leto storage. `Hephaestus` requires an acquired
provider and device-resident operands. Selection occurs once per plan; the hot
axis loops remain statically dispatched. Failure to acquire or execute the
selected accelerator is surfaced and never changes the selection. Kwavers PSTD
reuses a prepared Hephaestus plan rather than retaining a private shader.

After provider parity is established, Apollo deletes its WGPU FFT plan, shaders,
and WGPU `FftBackend` implementation, and Kwavers deletes both its Apollo GPU
re-export and PSTD FFT shader/dispatcher. No forwarding wrappers or deprecated
aliases remain. Apollo keeps CPU FFT algorithms; Leto keeps CPU layout/storage;
Hephaestus keeps accelerator FFT execution.

## Alternatives

- Extend Apollo's WGPU backend to 1-D/2-D and call it from Kwavers. Rejected
  because a domain algorithm crate would continue to own the shared device
  provider and Kwavers's second implementation would remain tempting.
- Move all FFTs into Leto. Rejected because Leto is below Apollo and owns layout,
  not transform arithmetic or accelerator APIs.
- Keep the PSTD kernel as a consumer-owned specialized fast path. Rejected after
  the matched benchmark proved a workgroup-local requirement that the generic
  provider can represent as an internal axis strategy. The optimized kernel now
  lives under the same prepared rank-generic plan and conformance suite as the
  staged radix and Bluestein strategies.
- Permit Hephaestus-to-Leto fallback. Rejected because backend identity would
  not describe the provider that executed the transform and device residency
  would be lost silently.
- Expose separate public 1-D, 2-D, and 3-D plan types. Rejected because rank is
  structural variation represented by one const-generic plan and operation
  seam.

## Consequences

Hephaestus gains a public additive operation family and provider-owned scratch
state. The Apollo and Kwavers cutovers are breaking removals of their GPU FFT
surfaces, with in-repository callers migrated in the same campaign. Leto needs
no FFT API; any measured CPU layout or transfer gap is implemented there as an
array operation without importing transform concepts. WGPU is initially the
only native FFT provider, so selecting another Hephaestus device reports
unsupported capability until that provider implements this seam.

Ownership movement alone establishes no speed claim. The executable plan's
static resource delta is exact: direct fixed-buffer binding removes two `N`
element allocations (`2N * size_of::<T>()`) and four `N` element device copies
(`4N * size_of::<T>()`) per dispatch. Binary16 therefore halves plan workspace,
root-table, and warm-copy byte volume relative to binary32 at the same shape.
Each staged plan adds `M + log2(M) + 1` scalar values for the largest staged or
Bluestein radix length `M`; each fused plan adds 1,035 scalar values. A
Bluestein axis adds one scalar reciprocal to its direct-real factor buffer.
These immutable prepared tables replace per-butterfly trigonometric and
division work. The half-circle representation avoids another `M` scalar values
per staged plan.
The matched PSTD instrument holds the input/output addresses and prebuilt plans
fixed, records all six pairs into one device-resident timed command stream, and
reports Criterion confidence intervals. Complete-step parity remains a consumer
cutover gate because the provider benchmark excludes Kwavers physics kernels.

## Verification

- Core planner tests cover ranks 0 through 4, singleton/nonzero shapes, checked
  products, dense and rejected strided layouts, split-component length and
  alias rules, and failure-atomic preparation.
- One generic WGPU conformance body instantiates `f32` and `eunomia::F16` across
  ranks one through three, fused and staged radix, Bluestein, and singleton
  axes. It checks direct analytical spectra, staged nontrivial Fourier modes,
  inverse round trips, NaN/infinity/zero behavior, and rejects an all-zero
  result under the normwise oracle.
  Binary16-specific tests cover a 3x3x3 Bluestein round trip, missing-feature
  and cross-device rejection with bitwise-unchanged operands, and stable
  prepared command/workspace identities through grouped warm dispatch.
  Apollo/Leto differential outputs remain a consumer-boundary cutover gate.
- FFT is the dependency-cycle exception recorded by the 2026-08-26 revision of
  ADR 0046: Hephaestus uses the analytical oracle here, while the Apollo/Leto
  differential test runs at the consumer boundary.
- Error bounds derive from scalar epsilon, transform length, and the radix or
  Bluestein operation depth; reordered floating-point results are not compared
  bitwise.
- Real-device WGPU tests prove one-, two-, and three-dimensional dispatch and
  large-prime Bluestein phase accuracy. Planner tests reject workgroup counts
  beyond the acquired device limit. Source/codegen audits prove one
  operation-boundary provider selection and no host fallback or per-axis
  capability probe.
- Prepared commands are immutable boxed slices and grouped warm encoding only
  validates device provenance before traversing them. Real-device tests keep
  command storage and workspace-buffer identities stable through dispatch;
  source inspection establishes that no Hephaestus-owned allocation, pipeline
  compilation, bind-group construction, host transfer, or device copy occurs
  on that path. This claim does not cover opaque allocations inside the driver.
- Synchronous readback completion state is retained at device construction.
  Tests prove that first and repeated readbacks allocate no new provider slot
  and that concurrent readbacks receive independent state. A whole-call global
  allocator census is not the ownership oracle: WGPU's one-shot encoder,
  submit, and mapping internals still allocate opaque host state. The Apollo
  phase census measured 99 allocations before removing the provider channel
  and 97 after it; provider counters and retained host-buffer identities cover
  the source-controlled lifecycle claim.
- Matched benchmarks compare Apollo/Leto and Hephaestus end to end at
  cache-/device-relevant shapes, reporting Criterion regression-slope time
  estimates and confidence intervals;
  no benchmark changes its workload or statistical acceptance to obtain a win.
  Small and medium instruments first check independent forward DFT bins with a
  standard `gamma(k)` bound derived from radix/Bluestein depth. The four-million
  element PSTD case uses an inverse round trip to avoid an O(N) trigonometric
  oracle in the benchmark binary; the same fused rank paths remain covered by
  direct analytical tests at smaller shapes, so an identity/no-op path cannot
  satisfy the combined validation.
- Residue scans prove Apollo and Kwavers retain no WGPU FFT shader, plan,
  dispatch helper, or compatibility re-export after cutover.
- Warning-denied checks, configured Nextest, doctests, SemVer checks,
  independent architecture review, and exact-head hosted provider and consumer
  CI pass before the campaign closes.
