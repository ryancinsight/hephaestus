# Checklist — hephaestus

Target version: 0.1.0. Sprint phase: Foundation → Execution.
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
