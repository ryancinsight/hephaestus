# Checklist — hephaestus

Target version: 0.6.2 (bumped; CHANGELOG synced). Sprint phase: Execution.
Phase 1 COMPLETE. Phase 2 gating ADR ACCEPTED (`docs/adr/0001-cuda-backend.md`
— cuda-oxide device substrate + cutile kernel authoring, SoC boundary,
no-toolkit-to-compile, differential parity vs CPU and wgpu). Next concrete
increment: `hephaestus-cuda` crate, stage 1 — device substrate on cuda-oxide
(acquisition, typed buffers, transfers) with skip-without-driver contract tests.

## 0.6.2 adaptive transient pools [patch]
- [x] Changed `BoundedBufferPool::recycle` to evict oldest retained buffers
  when the count cap is full, then enforce the byte cap.
- [x] Added regression coverage for full-pool small-buffer pollution and a
  zero-count invariant.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (26 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 174,145
  ns/iter; caller-owned 95,335 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.1 bounded transient pools [patch]
- [x] Added `infrastructure::pool::BoundedBufferPool` with retained-buffer
  count and byte caps.
- [x] Routed staging and uniform pools through the bounded pool while keeping
  existing WGPU buffer reuse semantics.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`;
  `cargo fmt --check`; `cargo check --workspace --locked`; `cargo clippy
  --workspace --all-targets --locked -- -D warnings`; `cargo nextest run
  --workspace --locked` (24 passed); `cargo test --doc --workspace --locked`;
  `cargo doc --workspace --no-deps --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo bench --bench elementwise_into
  --locked` on real adapter (allocating 272,830 ns/iter; caller-owned
  132,710 ns/iter for 1,048,576 elements, 20 iterations); `git diff --check`.
  Evidence tier: type-level ownership plus value-semantic unit/contract tests
  and empirical benchmark.

## 0.6.0 caller-owned contiguous elementwise [minor]
- [x] Added `binary_elementwise_into`, `unary_elementwise_into`, and
  `scalar_elementwise_into` for caller-owned output buffers and `BlockWidth`
  selection.
- [x] Routed allocating contiguous elementwise APIs through the caller-owned
  implementations; scalar dispatch now uses the uniform pool.
- [x] Consolidated pipeline-cache creation into `application::pipeline` for
  contiguous elementwise, strided elementwise, and reduction kernels.
- Evidence: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (22 passed);
  `cargo test --doc --workspace --locked`; `cargo doc --workspace --no-deps
  --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter
  (allocating 291,410 ns/iter; caller-owned 66,350 ns/iter for 1,048,576
  elements, 20 iterations). Evidence tier: value-semantic differential tests
  and empirical benchmark, not a stored Criterion regression baseline.

## Default provider feature contract [patch]
- [x] Added default `parallel` and `mnemosyne-memory` feature markers to
  `hephaestus-core` and `hephaestus-wgpu`.
- Evidence: `cargo metadata --no-deps --locked --format-version 1`; full Atlas
  feature-policy metadata audit; `cargo check --workspace --offline`;
  `cargo test --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo doc --workspace --no-deps --locked`;
  `git diff --check`.

## 0.3.1 uniform pooling + CUDA ADR [patch]
- [x] Pooled strided meta uniforms (queue-ordered write_buffer reuse);
  one fewer buffer allocation per dispatch. 17 tests green on hardware.
- [x] ADR 0001 accepted (Phase 2 gate).

## 0.3.0 strided unary/scalar + consolidation [minor]
- [x] Shared strided core (SSOT): `StridedMeta` packing, WGSL Meta/decode
  fragments, `cached_pipeline`, `encode_strided` serve all strided kernels.
- [x] `unary_elementwise_strided_into` (broadcast + caller-owned output).
- [x] `scalar_elementwise_strided_into` — zero new kernels (one-element
  operand at all-zero strides through the binary kernel).
- [x] Tests: unary transposed sqrt, unary broadcast neg, scalar/binary
  equivalence over a transposed view; 17 total on real hardware.
- [x] Gates: fmt, clippy `-D warnings`, test, doc — clean.

## 0.2.0 strided dispatch [minor]
- [x] `binary_elementwise_strided_into` over leto `Layout<N>` (rank ≤ 4
  compile-time cap, leto broadcast semantics, caller-owned output,
  aliasing/short-buffer rejection, packed 80-byte Meta uniform).
- [x] Differential strided suite (5) on real hardware; 14 tests total.
- [x] Gates: fmt, clippy `-D warnings`, test, doc — clean.

Previous sprint (0.1.0 scaffold) below.
In-flight item: none. Next concrete increment: strided-layout-aware dispatch (backlog Phase 1).

## 0.1.0 scaffold [arch]
- [x] Workspace: `hephaestus-core` (no GPU deps, `#![forbid(unsafe_code)]`,
  `#![deny(missing_docs)]`) + `hephaestus-wgpu` (wgpu 26).
- [x] `ComputeDevice` seam with GAT `Buffer<T: Pod>`; `DeviceBuffer<T>`;
  distinct error variants (adapter/device/length/dispatch/transfer).
- [x] `WgpuDevice` acquisition (default + custom limits) — single authoritative
  copy of the logic formerly duplicated in apollo-wgpu-helpers.
- [x] `WgpuBuffer<T>`: PhantomData-typed, padded allocation, `raw()` escape
  hatch for consumer-owned pipelines.
- [x] Upload (`create_buffer_init`), zeroed alloc, download (staging +
  map_async + poll), length-mismatch rejection before transfer.
- [x] `binary_elementwise::<Op, T>`: ZST `BinaryWgslOp` markers (Add/Sub/Mul),
  `WgslScalar` type-token substitution, arrayLength tail guard, partial
  trailing workgroup correct.
- [x] `unary_elementwise::<Op, T>`: ZST `UnaryWgslOp` markers (Exp/Ln/Sin/Cos/Sqrt/Abs/Neg/Recip) and shared WGSL template.
- [x] `scalar_elementwise::<Op, T>`: uniform buffer binding and ZST-wrapped pipeline cache keys.
- [x] `reduction::<Op, T>`: ZST `ReductionWgslOp` markers (Sum/Min/Max), multi-pass tree reduction, and type-safe `ReductionIdentity` mapping.
- [x] Contract tests (9): round-trip values, length rejection (download + dispatch), add/mul/unary/scalar/reduction vs CPU reference.
  Verified on real adapter hardware; environment-gated skip otherwise.
- [x] Gates: `cargo fmt --check`, `clippy --all-targets -- -D warnings`,
  `cargo test`, `cargo doc --no-deps` — all clean.
- [x] Pushed to GitHub; apollo delegation integration (see backlog Phase 4).
