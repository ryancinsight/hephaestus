# ADR 0053: Own multidimensional accelerator FFTs in Hephaestus

- Status: Accepted
- Date: 2026-08-26
- Revision 2026-08-26: The executable WGPU increment prebinds pipelines,
  immutable parameter buffers, bind groups, and dispatch grids during plan
  creation. The provider-neutral `FftOps::encode_fft` records into a
  caller-owned command stream, so composed consumers allocate no FFT-owned
  transient resources and can submit the transform with adjacent work.
  Bluestein phases are reduced modulo `2N` in integer arithmetic, evaluated in
  `f64`, narrowed once, and uploaded as direct complex factors. Benchmark
  submission and readback use explicit deadlines, and the required-device CI
  job executes the benchmark binary under a 60-second process bound.
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
existing provider command stream. The prepared operation remains bound to the
validated operand storage and shape, matching the existing Hephaestus
prepared-operation model.

`hephaestus-wgpu` is the first executable provider. It consolidates Apollo's
general radix/Bluestein implementation and Kwavers's PSTD requirements into one
rank-generic axis-pass plan. One-dimensional and two-dimensional transforms are
not wrappers around a three-dimensional public API: all ranks instantiate the
same axis planner, and only their actual axes are dispatched. All axis passes
are encoded into one command stream submission. Bluestein coefficient
preparation is Hephaestus-owned; integer modulo reduction bounds the quadratic
phase before `f64` evaluation and one `f32` narrowing. It cannot call Apollo or
silently execute the requested transform on the host. Metal continues to use
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
- Keep the PSTD kernel as a specialized fast path. Rejected unless a matched
  benchmark later proves a workload contract the generic provider cannot
  represent; retaining it now creates two authorities before such evidence.
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

No speed or memory improvement is claimed from ownership movement alone.
Performance claims require identical input/output addresses, prebuilt plans,
device-resident timed regions, allocation counts, and confidence intervals.

## Verification

- Core planner tests cover ranks 0 through 4, singleton/nonzero shapes, checked
  products, dense and rejected strided layouts, split-component length and
  alias rules, and failure-atomic preparation.
- One generic conformance suite covers ranks one through three using impulses,
  constants, single Fourier modes, inverse round trips, and Apollo/Leto
  differential outputs. It includes power-of-two and prime/composite
  non-power-of-two axes.
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
- Allocation and preparation instrumentation proves zero host and device
  allocation plus zero pipeline compilation during repeated dispatch. Buffer
  identities remain stable across iterations.
- Matched benchmarks compare Apollo/Leto and Hephaestus end to end at
  cache-/device-relevant shapes, reporting medians and confidence intervals;
  no benchmark changes its workload or statistical acceptance to obtain a win.
  Each instrument first checks independent forward DFT bins with a standard
  `gamma(k)` bound derived from radix/Bluestein depth, so an identity/no-op path
  cannot satisfy validation.
- Residue scans prove Apollo and Kwavers retain no WGPU FFT shader, plan,
  dispatch helper, or compatibility re-export after cutover.
- Warning-denied checks, configured Nextest, doctests, SemVer checks,
  independent architecture review, and exact-head hosted provider and consumer
  CI pass before the campaign closes.
