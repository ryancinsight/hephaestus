# macOS Metal verification checklist

This scoped acceptance checklist records macOS verification. The project
queue, priorities, and remaining hosted acceptance live in backlog.md.

Target version: **0.20.4**. Change class: **[patch]**. Phase: **Closure**.

- [x] [patch] Audit ADR 0047, Metal acquisition, operation inventory, and CI.
- [x] [patch] Refresh the task branch from current upstream and validate the
  standalone locked dependency graph on required native Metal.
- [x] [patch] Add direct acquisition, optional-feature, capacity-rejection,
  bounded-enumeration, and exact integer-dispatch contracts with tracing spans.
- [x] [patch] Configure the full canonical provider suite in macOS CI and trigger
  that lane on provider, conformance, facade, and verification-policy changes.
- [x] [patch] Verify minimal/default feature surfaces, the facade, strict
  Clippy, required-device Nextest, doctests, and warning-denied rustdoc.
- [x] [patch] Align feature-gated contract modules and the independent case
  inventory with all four decomposition/sparse feature combinations.
- [x] [patch] Exercise unsupported native precision through real fusion,
  stateful-update, and cross-entropy rejection; verify every operand survives.
- [x] [patch] Partition the three implementation-bearing manifests and two
  utility modules into responsibility-specific leaves; preserve mathematical
  operation order and effective device/infrastructure visibility.
- [x] [patch] Compare the local Hermes and Hephaestus providers in one consumer:
  native NEON and Metal both reduce 515 exact dyadic inputs to -37.625.
- [x] [patch] Fix Python embedding/extension link separation; retain both Rust
  binding tests, execute the WGPU binding lifecycle, and import a real extension
  whose Metal add/sum/min/max results match exact dyadic values.
- [x] [patch] Restore macOS CUDA-absence classification from native dyld
  diagnostics; verify real missing, invalid, symbol-incomplete, and dependent
  libraries, and confirm the required-CUDA flag still rejects absent hardware.
- [x] [patch] Close the fixed conversion budget with checked word/address,
  native-precision count, lossless pointer, and integer duration conversions;
  keep every fixture range, value oracle, tolerance, and workload.
- [x] [patch] Integrate upstream `cf811ae9`; partition its new utility modules
  into typed dispatch and host decomposition type/solve leaves, with direct
  device uploads and every existing numerical oracle retained.
- [x] [patch] Replace 64 moved upstream casts; preserve native probe-input bits
  and reference results, pointer identity, grids, and Python integers. Validate
  CUDA argument word boundaries before dispatch; pass all 780 workspace tests.
- [ ] [patch] Pass the final exact-head hosted Metal/provider gates and merge.
- [x] [patch] Preserve standalone Git dependency identities in Cargo.lock;
  synchronize version, README, backlog, audit, and changelog evidence.

Evidence (2026-10-13, aarch64 macOS, Rust 1.95.0, standalone `--locked`
dependency graph): 148 default WGPU/Metal, 82 minimal WGPU, 85
decomposition-only WGPU, 82 sparse-only WGPU, and 341 core/conformance/host
Nextest tests pass with zero skips; four doctests pass. Both Clippy surfaces
pass with warnings denied, and rustdoc passes with `RUSTDOCFLAGS='-D warnings'`.
The consolidated integration suites retain 191 default, 142 minimal, 189
decomposition-only, and 147 sparse-only named cases. The paired local-provider
consumer passes on Rust 1.97.0 with unchanged input readback.
Python validation on Rust 1.97.0, CPython 3.14.7 and NumPy 2.4.5 passes strict
Clippy and both Rust binding test cases (two tests). The WGPU lifecycle test
executes all three binding clauses; the CUDA test has no hardware evidence on
this Mac. A manual extension build under `PYO3_BUILD_EXTENSION_MODULE=1` imports
and computes exact 515-element Metal results; its linkage contains no Python
interpreter library.
The initial Rust 1.97.0 workspace run passes 777 tests with zero Nextest skips;
after upstream conversion closure, all 780 pass with the three new contracts.
Nine focused driver/compiler loader contracts pass; the dependency fixture
computes `x*x + x + 1` as 7, 13 and 31 before its owned dependency is removed.
CUDA hardware clauses remain unavailable here; these counts establish portable
Mac behavior and Metal execution, not CUDA hardware evidence.

Residual scope: hosted execution of the changed workflow remains unmeasured.
Intel hardware is not available in this verification environment. Native-kernel
and performance gaps in documented host-delegated WGPU families remain on the
existing roadmap, as does the separate ADR 0047 consumer migration. Existing
`HEPH-KERNEL-TAIL-ACCURACY` and `HEPH-RAY-INTEGRAL-DIRECTION-CONTRACT` items also
remain open; passing the current suite does not close those numerical gaps.

Acceptance: successful Metal acquisition retains `Backend::Metal`; all
advertised WGPU operation contracts pass through that adapter. Incompatible
limits and missing required devices fail. The existing 60-second per-test
timeout remains the verification bound.

The complete workspace doctest run includes the Python binding under default
embedding linkage and passes six executable examples with warnings denied.
