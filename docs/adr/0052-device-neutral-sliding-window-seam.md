# ADR 0052 — Device-neutral sliding-window seam

- Status: Proposed
- Date: 2026-08-18
- Refs: atlas `backlog.md#atlas-coeus-backend-045`; coeus ADR 0066
  (provider-owned dense product bridge, the matmul precedent); ADR 0044
  (the `DenseProductOps` seam this one is modelled on); ADR 0046 (host
  reference device and the Leto implementor); ADR 0041 (conformance crate).

## Context

Coeus closed its matmul fork onto `DenseProductOps` (coeus `afc1a7ce`,
coeus ADR 0066). Two op families could not follow: pooling and unfold/fold.
`coeus-cuda` and `coeus-wgpu` each carry their own, and the recorded reason
was that `hephaestus-core` has no pooling or sliding-window device trait.

Matmul was completable because `DenseProductOps` (ADR 0044) already existed
with all four vendor crates implementing it; Hephaestus needed no change at
all. That precondition does not hold here, and the gap is deeper than one
missing trait.

### What the two dialects actually share

A file-level comparison of the Coeus CUDA and WGPU trees was run before
proposing a seam, because a seam forced onto structurally different kernels
is worse than the duplication it removes. They are not structurally
different. The device code is a transliteration:

- **Pooling forward** is one invocation per output element, flat 1-D index,
  the same `% /` decomposition into `(n, c, oh, ow)`, the same nested
  window loop with per-tap bounds checks, one store. `coeus-cuda`
  `src/kernels/pool/max.rs:90-136` against `coeus-wgpu`
  `src/kernels/pool/max.rs:89-142` agree down to variable names (`temp1`,
  `has_val`, `h_in_limit`) and the tap formula. Average pooling likewise
  (`avg.rs:79-121` / `:84-132`), both counting in-bounds taps rather than
  dividing by a fixed `k*k`.
- **Pooling backward** is the same strategy in both, which is the result
  that matters most: neither stores argmax indices and neither scatters.
  Both launch one invocation per *input* element and gather over the
  inverse-stride set, recomputing the window argmax from the input
  (`max.rs:303-349` / `:320-366`). WGSL's restricted atomics did not force
  a divergence, because the CUDA side independently chose the atomics-free
  gather. There is no `atomicAdd` anywhere in the CUDA pooling tree.
- **Unfold/fold** is the same again: gather-per-output-element in both
  directions, no atomics, identical index algebra, and the same non-obvious
  recovery of input height by division rather than by parameter
  (`coeus-cuda` `src/kernels/unfold_fold/source.cu:153` against
  `coeus-wgpu` `src/kernels/unfold_fold.rs:134`).
- **Launch geometry** is `ceil(n/256)` over a 1-D grid in both dialects,
  from `launch_grid_size` (`coeus-cuda/src/kernels/validation.rs:96-105`)
  and `checked_workgroup_count` (`coeus-wgpu/src/backend/error.rs:190-202`).
- **Output-shape derivation** is algebraically identical:
  `(in + 2p - d(k-1) - 1)/s + 1` (`coeus-cuda`
  `src/kernels/unfold_fold/validation.rs:47-64`, `coeus-wgpu`
  `src/kernels/unfold_fold/validation.rs:162-203`).

Neither dialect uses shared-memory tiling, cooperative reduction, or
multi-pass anywhere in these families. The two places pool dialects usually
fork — backward accumulation strategy and index storage — do not fork here.

So the seam is real. The board's recorded primitive ("a 1-D launch over
`numel(output)` at workgroup 256 taking layout descriptors plus 4 or 9 `u32`
params") is confirmed, though it understates the case: the shared part is
not only the launch convention but the whole host-side geometry, validation,
and shape derivation, roughly 35% of each tree computing the same numbers
twice.

### Divergences that are contract decisions, not obstacles

Three differences must be settled by the seam rather than preserved:

1. **Empty operands.** CUDA rejects any zero extent
   (`unfold_fold/validation.rs:84-89`); WGPU returns success
   (`unfold_fold.rs:231-233`). Same call, different result today.
2. **Validation depth is one-sided.** CUDA's pooling runs
   `pool_index_arithmetic_is_valid`, `pool_prefix_matches`,
   `pool_shapes_match`, and a storage-bounds check
   (`coeus-cuda/src/kernels/pool/validation.rs:6-99`) that WGPU has no
   counterpart for (`coeus-wgpu/src/kernels/pool/validation.rs:5-29`).
   WGPU additionally requires output contiguity for unfold/fold, which CUDA
   does not check. Each gap is a latent defect in the other backend.
3. **Scalar sets do not overlap** beyond `{f32, i32}`: CUDA carries
   f64/f16/bf16, WGPU carries u32. The seam must be scalar-generic with a
   per-backend narrowing bound, as `DenseProductOps` already is, not a
   fixed common set.

### Why this is not one increment

`hephaestus-core` has the right shape to copy: `ConvolutionPlan<S>`
(`domain/convolution/plan.rs`) already carries batch, channel and spatial
extents, validated parameters, output element count, largest physical
offset, and an address-limit check. A `PoolingPlan<S>` mirrors it directly.

The obstacle is below that. `DenseProductOps` was landable in Coeus because
four vendor implementations already existed upstream and had passed the
conformance suite. For pooling, **nothing exists upstream at any level**:

- `leto-ops` has no pooling and no unfold/fold. It has convolution
  (`application/convolution/`) but the pooling family is absent, so the
  CPU reference does not live where the stack expects it. It does exist,
  one layer too far downstream: `coeus-ops` carries a complete generic CPU
  implementation in `src/backend_ops/cpu_impl/` — `impls/pool.rs`,
  `pool/pool1d.rs`, `unfold_fold.rs`, `impls/unfold_fold.rs`, roughly 900
  lines, already written as `impl<T: Scalar + leto_ops::Scalar, B:
  CpuBackend>`. It is also the oracle the CUDA parity suites compare
  against (`coeus-cuda/tests/cuda/parity/pooling.rs`). Stage 1 is therefore
  a promotion of existing, tested code into its owning repository, not a
  new implementation.
- Consequently `hephaestus-host` — the reference substrate ADR 0046
  established precisely so seams get a testable implementor, by adapting
  leto-ops entry points — has nothing to adapt.
- The four vendor crates have no pooling kernels; the device source lives in
  Coeus and would have to be re-homed and re-expressed against the seam.

A seam added to `hephaestus-core` today would therefore have no implementor
at all, which is speculative scaffolding, and the Coeus collapse cannot
precede the vendor implementations without leaving exactly the half-migrated
family the standards forbid.

## Decision

Adopt the seam, and stage it in dependency order. The seam is:

- `PoolingParameters<D>` in **leto**, a sibling of `ConvolutionParameters<D>`
  carrying the per-axis window extent that convolution takes from its weight
  tensor instead.
- `PoolingPlan<S>` and `SlidingWindowPlan<S>` in **hephaestus-core**,
  mirroring `ConvolutionPlan<S>`: extents, derived output shape, element
  count, max physical offset, `validate_address_limit`. This is where the
  duplicated shape math consolidates, and it is pure host arithmetic, so it
  is fully testable without a device.
- `PoolingOps<D, T>` and `SlidingWindowOps<D, T>` in **hephaestus-core**,
  shaped like `ConvolutionOps` (prepare/dispatch split, associated prepared
  types, scalar bound narrowed per implementor) rather than like the
  single-call `DenseProductOps`, because these families carry compiled
  pipelines and per-operand validation exactly as convolution does.

The contract decisions the seam settles: empty operands succeed as a no-op;
the union of both validation suites applies to every backend; scalar support
is per-implementor.

Staging, each stage a complete vertical increment:

1. **leto-ops CPU pooling and unfold/fold**, with `PoolingParameters`,
   promoted from `coeus-ops`'s `cpu_impl` tree rather than written fresh;
   Coeus then consumes it instead of carrying it. Establishes the reference
   semantics and the differential oracle.
2. **hephaestus-core plan and seam traits**, with the plan's shape
   arithmetic unit-tested against the leto reference.
3. **`hephaestus-host` implementor** adapting stage 1 (ADR 0046 pattern),
   plus the **conformance suite** — the first stage whose correctness is
   fully verifiable on a host with no GPU.
4. **Vendor implementations** (cuda, wgpu, metal, rocm), each verified by
   the stage-3 conformance suite on GPU-capable CI.
5. **Coeus collapse**: `PoolProvider`/`PoolBackend` and one generic
   dispatch mirroring the matmul seam, vendor kernels deleted, not wrapped.

Stages 1-3 are verifiable on a CPU-only host. Stages 4 and 5 are not.

## Alternatives

- **Add the seam to `hephaestus-core` now and let Coeus implement it
  directly from its existing kernels.** Rejected: it inverts the ownership
  the seam exists to establish. The vendor kernels would stay in the
  consumer, so Metal and ROCm would remain unserved and the fork would
  persist behind a trait — the shape of a compatibility layer, not a
  migration.
- **Move the Coeus CUDA and WGSL kernel text upstream verbatim in one
  change.** Rejected on evidence grounds, not effort. It is roughly 2.8k
  lines of pooling and 0.9-1.0k of unfold/fold per vendor crate, and the
  development host has no CUDA device and no usable WGPU adapter (the
  matmul parity suites are already among 116 `AdapterUnavailable`
  failures). The change would delete working, shipping kernels in favour of
  re-homed ones that could be neither executed nor, for Metal and ROCm,
  compiled. The duplication is cheaper than an unverifiable migration.
- **Implement the CPU reference in `hephaestus-host` rather than
  `leto-ops`.** Rejected: ADR 0046 §3 fixes the host ops as adapters over
  leto-ops entry points so that Leto is the role-trait implementor per ADR
  0039 §3. Putting array compute in `hephaestus-host` would place domain
  logic in the adapter and leave the upstream gap unfilled.
- **Generalize `ConvolutionOps` to cover pooling instead of adding a
  sibling.** Rejected for now: convolution's operands carry weight and bias
  tensors and its parameters take the window extent from the weight shape.
  Folding a weightless, window-parameterized family into it would widen
  every existing implementor's contract. Revisit once both plans exist and
  the shared part is visible in code rather than assumed.
- **Keep the families forked and close the item.** Rejected: the
  duplication is real and now precisely characterized, and the seam is
  sound. It is staged, not abandoned.

## Consequences

`HephaestusBackend<P>` continues not to satisfy Coeus's `BackendOps<f32>`;
Metal and ROCm remain partial for these two families until stage 4. The
duplication in `coeus-cuda` and `coeus-wgpu` persists meanwhile and is
accepted with this record as its justification.

Stages 1-3 change no vendor behaviour. Stage 4 changes it in three named
ways — empty-operand handling, validation depth, and error typing — and
each is a behaviour change that the conformance suite, not the developer
host, must confirm.

## Verification

Nothing in this ADR is implemented; it records the design and the staging.
The comparison it rests on is a read of the two Coeus trees at coeus
`79f05dfd`, cited inline by file and line above. No kernel was executed:
the host has no CUDA device and no usable WGPU adapter.

## Revisit trigger

Any of: leto-ops gains the pooling family (unblocks stage 2); a
GPU-capable CI lane becomes available (unblocks stage 4 verification and
also the still-unverified matmul parity from coeus ADR 0066); or a third
backend needs pooling, which would raise the cost of the fork above the
cost of the migration.
