# Native macOS publication closure

Change class: [patch]. This partition enforces the existing ADR 0042/0047
boundaries without changing a public trait, operation, scalar, or backend.

The publication scan identifies implementation-bearing module manifests in
decomposition conformance, WGPU device infrastructure, and shared CTC kernel
dispatch. It also identifies decomposition `common` and stream `helpers`
modules whose names conceal their mathematical or binding responsibilities.

The selected partition gives each responsibility one leaf: decomposition
contract aggregation, identity factorizations, and its derived error bound;
WGPU device state, cache keys, mapping lifetime, pool retention, and bounded
waits; CTC compiled step dispatch; and stream storage binding layout. Module
manifests contain declarations and canonical exports. Every old internal
utility module is removed and its callers use the new responsibility directly.

The invariant is semantic identity: the same decomposition clauses run in
the same order, every CTC entry retains its source, storage indices, parameter
type and dispatch grid, and WGPU retains identical device fields, reference
ownership, pool capacities, idle-decay and wait deadlines. No new dispatch,
allocation, synchronization, or mathematical algorithm is introduced.
Moved private fields use the existing device module as their visibility scope;
the partition does not expose device storage outside that boundary.

Alternatives rejected: accepting a higher debt baseline leaves the source
defect intact; moving implementations into another manifest recreates it;
forwarding wrappers duplicate the authoritative surface. Ordinary exports of
the sole canonical leaf definition preserve the established public module path.

Verification compares moved bodies with the pre-partition source, runs strict
diagnostics and documentation, and executes all four WGPU feature surfaces,
Metal acquisition/operation contracts, and core/conformance/host tests with
the existing 60-second per-test bound. Mapping, retention, error, and numerical
failures keep their existing tests and value oracles.

The final conversion gate identifies inherited bare casts above its fixed
rows, including casts newly exposed by the partition. Replace whole operation
families' conversions with checked address/word construction, lossless pointer
construction, and Eunomia's native-precision count conversion. Signed device
address rejection is equivalent to failed `i32::try_from` for the nonnegative
span/product. The CTC constructor already proves its batch and class words fit
`u32`; tile constants and device words fit supported host indices. Float fixture
counts retain the same rounded binary32 values and arithmetic order; no input
range, tolerance, or workload changes. Prepared-buffer pointers retain exact
identity through `std::ptr::from_ref`.

Retention compares integer tick differences in Duration's u128 millisecond
unit, so deadlines cannot truncate; stored u64 epochs use checked construction.
No mathematical kernel widens its scalar arithmetic. Strict conversion counts
fall from 50/48/37/169/319 to 30/24/19/162/298 for conformance/core/host/ROCm/WGPU,
meeting the existing 31/24/20/163/303 rows without changing a baseline.

The paired consumer uses `x_i = ((i mod 32) - 16) / 8` for `0 <= i < 515`.
Each complete cycle sums to -2; sixteen cycles plus the final three terms
sum to -37.625. All partial sums are integer multiples of 1/8 with integer
magnitudes bounded by `515 * 16 < 2^24`, so every binary32 reduction order is
exact. Hermes' native NEON and Hephaestus' Metal sum must equal that value.
The imported extension doubles the same operands: sum -75.25, minimum -4,
maximum 3.75, with the original operand unchanged.

The full publication gate also builds the Python binding's real Rust tests.
The inherited workspace-wide PyO3 `extension-module` feature suppresses the
interpreter library those executables need. Remove that deprecated feature;
default Cargo tests use PyO3's discovered interpreter link, while Maturin
>= 1.9.4 sets `PYO3_BUILD_EXTENSION_MODULE` for wheel artifacts. PyO3's build
helper remains the sole authority for target-specific cdylib flags. Both native
binding tests and an actual extension import/value check verify these distinct
link contracts. No binding test or harness is disabled. See
[PyO3's documented root cause](https://pyo3.rs/v0.29.0/faq.html#i-cant-run-cargo-test-or-i-cant-build-in-a-cargo-workspace-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror).

CUDA acquisition already specifies adapter absence for a missing driver and
device faults for a present but unusable driver. The ELF-only loader diagnostic
classifier violates that contract on macOS. The dyld branch accepts only a
direct-image header naming the requested image and a nonempty sequence of
requested-file search attempts whose outcomes all state physical/cache absence.
Each iteration consumes one attempt; unknown diagnostics and dependent-image
failures reject the absence classification. The branch uses dyld's own search
evidence because a system image can live solely in its shared cache.

Native loader tests cover an absent image, a plain file, a loaded system image
without CUDA/NVRTC exports, and two compiled input-sensitive C libraries. The
last fixture computes `x*x + x + 1` for 2/3/5 before removing its dependency;
reloading the present parent must preserve a device fault with both image names.
The physical CUDA transfer test retains all value assertions and still fails
when hardware is required. The full workspace verifies this portability fix
together with the native Metal surface.

Current upstream integration (`cf811ae9`) introduces two further utility
modules. Partition typed-elementwise execution into comparison and arithmetic
dispatch leaves, retaining each operation, layout, upload, allocation, dispatch,
and readback unchanged. Host decomposition separates its zero-sized operation
type from the Leto solve bridge. Callers upload through `ComputeDevice` directly;
the internal upload-only forwarding function is removed. Public definitions,
solve order, buffers, errors, numerical inputs, and value oracles remain identical.
The inherited QR routing optimization and source partitions remain in the merge.

The upstream merge also moves 64 bare conversion sites into new leaf files.
CUDA launch arguments use `ptr::from_mut(...).cast()`, preserving each stack
argument's address and lifetime. Matmul workgroup counts derive from dimensions
already checked by `GpuMatrixLayout`; `ceil(n/16)` has the same value at the
validated word width. Batch counts and reduction lengths must fit the kernels'
unsigned 32-bit arguments before any dispatch. Batch chunks satisfy
`offset + chunk <= count <= u32::MAX`; reduction counts decrease by ceiling
division with the validated block width. Neither argument can truncate.

Canary inputs use one native-precision generator. Integer samples, dyadic
fractions, 1/10, and each format's E constant retain every original input bit;
the test checks both operand arrays for all four builtins at both precisions.
The associated `Reference` type declares the binary64 CPU truth used to grade
lowering accuracy. Generic reference evaluation retains the native libm result
bits; live precision probes still execute on Metal. GPU arithmetic does not
widen. Cache tags preserve the original builtin discriminants. Python returns
native-width permutation indices as the same exact Python integers, avoiding
an unnecessary intermediate widening. Existing fixtures, tolerances, workloads,
errors, and public numerical APIs remain unchanged. Oversized CUDA counts now
return a typed dispatch failure instead of truncating their kernel arguments.

Hosted Metal run `38458373262` exposes an empirical canary cutoff that rejects
an error allowed by the portable builtin accuracy model. Replace all fixed
precision cuts with one operation/precision calibration owner. The co-located
`canary/accuracy.rs` proof derives the fixed-probe ULP budgets and the comparison
bound `(gpu_error + reference_error) / (1 - reference_error)`. Native probe
inputs, shader arithmetic, reported errors, and numerical-kernel value oracles
remain unchanged. Binary64 uses an explicit width-scaled diagnostic model;
WGSL supplies no binary64 conformance contract, and illegal lowerings still
reject before shader construction. Exact-zero reference outputs require exact
zero GPU results; nonzero and nonfinite results cannot hide behind a denominator
floor. Required canary acquisition selects Metal and preserves every device fault.

Integer-rational tests decode the measured binary64 bounds and compare them
with exact numerator/denominator products in at most 107 bits. They prove
outward enclosure and at most two adjacent steps of rounding headroom. Native,
single-grade, invalid-error, and rejection boundaries retain value assertions.
See the [portable WGSL accuracy rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy).
