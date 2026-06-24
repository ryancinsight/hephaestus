# Gap Audit - hephaestus

This register is the SSOT for hephaestus's known limitations. Items are grouped
by actionability so an open defect is never confused with an intentional
architectural decision or a tracked future-work item:

- **Resolved** — closed with evidence (kept for traceability).
- **Accepted design decisions** — intentional architecture, not defects.
- **Open future work** — tracked GPU-kernel / performance parity requiring
  `[major]` effort; the wrappers are API-complete and value-verified, the gap is
  native-kernel/performance parity, not correctness.
- **Environment / toolchain limitations** — blockers outside the source tree.

## Resolved

- [minor] WGPU `matrix_rank` ill-conditioned threshold contract is now pinned and
  documented. The kernel counts pivots greater than
  `relative_tolerance * max(abs(matrix))`; this is a row-reduction (pivot
  magnitude) criterion rather than Leto's SVD-spectrum criterion, so the two can
  diverge on ill-conditioned inputs where pivot magnitudes and singular values
  differ. The boundary behaviour — same near-singular matrix is full-rank or
  rank-deficient depending purely on the relative threshold, and agrees with Leto
  when pivot magnitudes equal singular values (diagonal/orthogonally-scaled
  inputs) — is documented on `matrix_rank_with_tolerance` and pinned by
  `matrix_rank_relative_tolerance_is_the_discriminator`. Evidence tier:
  value-semantic threshold-boundary contract test plus Leto differential.
- [minor] WGPU `det` ill-conditioned contract is now pinned and documented. `det`
  passes `relative_tolerance == 0`, so only an exactly-zero pivot collapses the
  determinant to zero; a near-singular matrix returns its small, nonzero pivot
  product (no determinant-tolerance zeroing), which can diverge from an
  SVD/tolerance-based determinant on ill-conditioned inputs. Documented on `det`
  and pinned by `det_of_near_singular_triangular_is_exact_pivot_product` (exact
  analytical pivot product `2·3·δ` on an upper-triangular input, plus Leto
  differential). Evidence tier: analytically-derived value-semantic test.
- [patch] Hephaestus WGPU launch planning uses Mnemosyne `KernelResourceBudget`
  and Moirai GPU `plan_launch` through Moirai's planner-only feature set. The
  prior duplicate-WGPU risk is closed: Hephaestus depends on `moirai-gpu` with
  default features disabled, so `moirai-gpu` no longer pulls `wgpu 0.19` into the
  Hephaestus graph. Evidence tier: dependency-tree verification and package
  checks.
- [patch] Python binding tests previously hung after the RNG binding test body
  completed because transient WGPU staging/uniform buffers could remain retained
  across short-lived host-runtime teardown. `PyDevice` now drains the bounded
  WGPU transient pools on drop. Evidence tier: root-cause diagnostic run showing
  the test body completed before process hang, Python package nextest, and full
  workspace nextest.
- [patch] WGPU staging-allocator registry (`WGPU_MAPPED_BUFFERS`) contention
  audit (2026-06-23). The HostPinned alloc/upload paths resolve a Mnemosyne
  sub-allocated pointer to its containing wgpu mapped block. This previously held
  the global registry lock across an `O(n)` `.iter().find()` linear scan over
  every live mapping (`O(n²)` over a staging-heavy workload). Replaced the
  `HashMap` with a base-address-keyed `BTreeMap` and consolidated both sites into
  one `resolve_mapped_buffer` helper doing an `O(log n)`
  `range(..=ptr).next_back()` containment query, shortening the lock-held section.
  Remaining bounded characteristic (not a defect): a single global `Mutex` still
  serializes the registry across threads; the lookup is now `O(log n)` and the
  `wgpu::Buffer` return is an `Arc` handle clone, so the critical section is
  minimal. The deeper lever, if a future staging-heavy multi-thread workload
  measures the global lock as hot, is a sharded registry keyed by address range —
  deferred until a workload demonstrates the need (no current consumer drives
  concurrent HostPinned staging). The registry and its descriptor are now
  `pub(crate)` (no external consumers) and the dead `usage` field is removed.
  Evidence tier: value-semantic placement/transfer contract tests + full
  workspace gate.

- [patch] Blocked-decomposition per-panel host-allocation churn (audit
  2026-06-23). The blocked Cholesky/LU/QR panel loops allocated a fresh host
  `Vec` every panel for each region download (`col_panel`/`row_panel`/`panel`,
  up to `b·n` ≈ 128 KiB/panel at n=512) plus per-panel scratch (LU `diag`, QR
  reflector-packing buffers) — `O(n/b)` host allocations per call on top of the
  already-pooled device buffers. Resolved: the region-download SSOT is now
  `download_matrix_region_compact_into(out: &mut Vec)` (reuses host capacity via
  `resize`); the dead returning-`Vec` `_reusable` wrapper is removed; each
  decomposition hoists its per-panel host scratch above the loop and refills it.
  The remaining blocked-decomposition perf gap is the host/device
  synchronization floor and native-kernel parity (tracked under Open future
  work), not host allocation. Evidence tier: cross-block-boundary value-semantic
  contract tests (blocked LU n=66, blocked Cholesky/QR across boundary) + full
  workspace gate.

## Accepted Design Decisions

- WGPU/Leto parity is complete for the current core array-operation slice:
  elementwise, strided elementwise, scalar elementwise, reductions, rank-2 axis
  reductions, rank-2 scans, matrix products, Kronecker product, matrix power,
  finite-`f32` matrix rank, finite-`f32` determinant, dot, trace, norms,
  Cholesky/LU/full-pivot LU/QR/column-pivoted-QR/SVD/bidiagonalization/Schur/Hessenberg/Bunch-Kaufman/UDU
  decomposition APIs, pseudoinverse and matrix exponential baseline wrappers,
  symmetric Jacobi eigen decomposition/eigenvalue APIs, and general eigenvalues
  for diagonal closed-form and nonsymmetric Leto-differential cases. Evidence
  tier: value-semantic contract tests against CPU references and Leto, plus
  comparative benchmark evidence in `benchmark_results.md`. (Per-operator
  GPU-kernel/performance parity for the host-delegated wrappers is tracked under
  Open future work.)
- [minor] Hermes integration is intentionally host-tier for Hephaestus:
  host-delegated parity wrappers call `leto-ops` with `simd` enabled, and Leto
  routes CPU hot loops through Hermes SIMD before Hephaestus uploads verified
  outputs into device buffers. Direct WGPU/CUDA kernel calls into Hermes are out
  of scope because Hermes owns CPU SIMD over host slices while Hephaestus owns GPU
  resource lifetimes and device-resident kernels. Evidence tier:
  dependency/implementation audit and ADR 0002.

## Open Future Work — GPU-kernel & performance parity

These surfaces are API-complete and value-verified against Leto/nalgebra; the
remaining gap is native-GPU-kernel and/or performance parity (`[major]` effort),
not correctness. Factorization/solve currently delegate to Leto on the host
before uploading device buffers.

- [minor] WGPU Cholesky/LU/QR provide device-resident factors and Leto-matching
  solve/inverse/determinant surfaces, but factorization delegates to Leto on the
  host before uploading the factors (API parity, not GPU-kernel parity). Evidence
  tier: implementation audit, value-semantic differential tests, and comparative
  benchmark rows.
- [minor] WGPU symmetric Jacobi eigen decomposition provides device-resident
  eigenvalues/eigenvectors, but the eigensolve delegates to Leto on the host
  before uploading the outputs (API parity, not GPU-kernel eigensolver parity).
  Evidence tier: value-semantic differential tests, non-symmetric rejection test,
  and comparative benchmark row.
- [minor] WGPU general eigenvalues are exported with complex device buffers and
  covered for diagonal, exact complex-pair blocks, triangular, structured
  nonsymmetric real-spectrum, dense `nalgebra` differential, symmetric-real, and
  rectangular-rejection cases (32x32 block-rotation benchmark against Leto and
  `nalgebra`). Remaining risk is API/performance parity only: the wrapper
  delegates to Leto on the host before uploading complex device buffers. Evidence
  tier: value-semantic closed-form, differential, invalid-input tests, and
  empirical benchmark row.
- [minor] WGPU pseudoinverse and matrix exponential have non-diagonal,
  rank-deficient, rectangular, nilpotent, skew-symmetric, general-matrix, and
  invalid-input contract coverage plus comparative benchmark rows. Remaining risk
  is performance/API parity only: both wrappers delegate to Leto on the host and
  upload device buffers. Evidence tier: value-semantic closed-form, Moore-Penrose
  algebraic, differential, invalid-input tests, and empirical benchmark rows.
- [minor] WGPU blocked Cholesky offloads the trailing SYRK update to a GPU
  kernel, but diagonal panel factorization and triangular panel solves remain
  CPU/Leto-backed. Current empirical row: 128x128 blocked Cholesky is slower than
  Leto and `nalgebra` on the local WGPU run. Evidence tier: value-semantic
  differential test across a block boundary and empirical benchmark row in
  `benchmark_results.md`.
- [minor] WGPU blocked LU and blocked QR have comparative benchmark rows. Blocked
  LU transfers are narrowed to the active diagonal-panel and trailing-submatrix
  regions. Blocked QR transfers compact trailing-column tiles per panel before
  GPU Householder application and uploads all panel Householder vectors in one
  packed buffer. The measured 66x66 blocked LU row remains slower than Leto and
  `nalgebra`; the 70x35 blocked QR row is much slower than Leto and `nalgebra`.
  Evidence tier: value-semantic blocked LU/QR tests plus empirical benchmark rows.
- [minor] Blocked decomposition synchronization profiling shows a material, noisy
  transfer/synchronization floor after the blocked LU region-transfer reduction,
  blocked QR compact-tile transfer reduction, and packed reflector upload.
  Timestamp queries measure the QR launch component directly: 32 separate
  reflector-equivalent compute passes previously totaled 155.2 µs on the local GPU
  timeline (3.4 µs median pass). The WGPU QR panel path now applies all panel
  reflectors in one compute pass per panel; the timestamp profile is 8.2 µs total
  (160 ns median), and the 70x35 blocked QR row measures 420.8 µs. The component
  profile measures the 70x35 CPU panel-factorization lower bound at 26.3 µs and
  the duplicated final Leto recompute at 11.5 µs, while the synthetic QR
  host/device synchronization floor remains 222.6 µs. The trailing-update kernel
  packs Householder vector offsets and beta coefficients into one reflector
  metadata buffer (two storage bindings → one). The 70x35 comparative row did not
  improve after this packing change: WGPU 480.8 µs vs Leto 14.9 µs and `nalgebra`
  10.0 µs. Remaining risk: blocked QR still trails Leto/`nalgebra`; the next
  measured lever is reducing the host/device synchronization count, not metadata
  buffer count or CPU panel arithmetic. Evidence tier: value-semantic blocked QR
  tests, empirical synchronization/component profiles, comparative benchmark, and
  GPU-timeline timestamp measurement in `benchmark_results.md`.
- [minor] WGPU CSR sparse storage uploads Leto CSR matrices into device-resident
  values plus one packed index buffer and executes SpMV/SpMM in WGSL without
  downloading operands to the host. The kernel layout stays within WGPU's portable
  four-storage-buffer limit, and dispatch sizing reuses the shared Mnemosyne/Moirai
  launch-planning helper. The focused sparse comparative harness validates WGPU
  outputs against Leto before timing: SpMV 1000x1000 CSR measured WGPU 100.888 µs
  vs Leto 1.280 µs; SpMM 1000x1000x128 measured WGPU 73.634 µs vs Leto 32.470 µs.
  Remaining risk: sparse performance parity is not achieved for either SpMV or
  SpMM on this run; no `ndarray`/`nalgebra` sparse comparator is recorded because
  the current Leto sparse API benchmark has no dense-library sparse equivalent in
  this harness. Evidence tier: static diagnostics, value-semantic WGPU sparse
  contract test, value-checked benchmark outputs, and empirical benchmark.

## Environment / Toolchain Limitations

- [minor] CUDA mirrors the current core operation and decomposition slice in the
  source tree and passes stub-mode verification. Real CUDA feature verification is
  still required on CUDA hardware/toolchain before claiming device-execution
  parity for the CUDA kernels. CUDA blocked Cholesky remains CUDA-feature gated and
  is not part of the default stub-mode claim. Evidence tier: static diagnostics and
  stub-mode contract tests. (Blocked on CUDA hardware availability.)
- [patch] Full workspace `--all-features` clippy is blocked by `cuda-bindings`
  requiring `CUDA_TOOLKIT_PATH`. The canonical clean local gate is therefore
  default-feature scoped (`cargo clippy --workspace --all-targets -- -D warnings`
  + `cargo nextest run --workspace`, which builds `hephaestus-cuda` in stub mode);
  the `--all-features` variant runs on CUDA-equipped CI. Evidence tier: static
  diagnostics from the attempted all-features gate; default-feature gates are
  clean.

## Next Increment

- Continue the parity audit at the next highest-risk Open future-work residual:
  reduce blocked QR host/device transfer and synchronization costs before
  replacing CPU panel arithmetic with native GPU kernels.
