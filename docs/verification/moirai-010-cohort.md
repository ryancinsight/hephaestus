# Moirai 0.10 synchronization provider continuation

This continues #497 at reserved 0.22.2. Owner `bb296d724b196e2939d6e68a066aaaf3c92a8194`
already contains current master `b03e91761cc929d046bd0d911bd573c699560d8e`.
Every kernel and test body remains identical to the owner head. The only Rust
edit scopes CUDA-only/decomposition-only imports to their existing feature guards.

The canonical Moirai-sync requirement advances from 0.9.3 to 0.10.0. Normal
Cargo selects Moirai `68466225913301d30281b134d42b17c604baf78e`, Leto 0.47.0
`b44b9f07b0fefab0f576ba971051514aca1c8ad6`, and current compatible canonical
numeric providers. No manifest revision pin or path override is introduced.
The existing Leto interval continues to admit both 0.46 and 0.47.

ARM64 Mac verification, with the committed canonical review graph:

- Rust 1.97: 444 core/host/conformance native tests pass, zero skips; workspace
  all-target strict Clippy, six workspace doctests and warning-denied docs pass.
- The facade contains no native cases. Its standalone Nextest selection reports
  exit 4 (zero tests); compilation succeeds. The complete CI host selection
  includes core/host cases and is checked separately without an empty-test override.
- Rust 1.95: all six existing Metal feature/facade checks and both strict Clippy
  modes pass. Actual required-device WGPU minimal tests pass 96 cases; default
  WGPU/Metal tests pass 184 cases, both with zero skips. Device doctests and
  warning-denied docs pass. Both HEPHAESTUS_*_REQUIRE_DEVICE flags equal 1 and
  WGPU_BACKEND equals metal. Existing serial groups and test budgets are unchanged.
- The artifact-budget check passes at its unchanged 1,000-line ceiling.

The final Leto revision changes documentation only relative to the device-tested
`b994b32c` source; its Rust and dependency bytes are identical. Native host checks
also execute after that canonical lock refresh. No numerical/backend operation
changes are part of this migration. Moirai-sync is used in private CUDA pipeline
cache fields; no new public synchronization type enters the facade.

Five exact-source portable patch API comparisons pass 223 checks each (30 tool
skips each) at `aecba2ce`. The subsequent CUDA import-only change is checked
with strict minimal/default Clippy and the unchanged CUDA native suites. The
minimal mode previously fails first on the production import and then on six
test imports; all seven imports now follow their existing cfg uses. Test bodies,
feature activation and dispatch remain unchanged. Final CUDA API/source and
hosted receipts are recorded separately. The final CUDA comparison at
`893276ca` passes all 223 patch checks (30 tool skips); the other four package
source trees remain identical to `aecba2ce`. Both CUDA modes pass strict Clippy
on Rust 1.95 and 1.97. Existing adapterless selections pass 139 minimal and
250 default cases; these are not NVIDIA hardware execution receipts. ROCm's production-feature/API check belongs to its existing
Linux SDK lane. This Mac provides no NVIDIA/AMD device execution evidence.
Provider-first default-branch and registry availability remains an integration
gate for unseeded consumers. Coeus Miri remains deferred and unverified.

Aequitas #141 proposes law 0.4.0 / binding 0.2.0 at `8c50883a`; ADR 0020,
owner #140 coordination and review/hosted/release acceptance remain pending.
This source retains the current coherent 0.3 intermediate graph. No generic
Quantity::powi or TaggedDimension caller occurs here; `Laplacian2DParams::new`
exposes Length<f32> at `domain/stencil.rs:32` and must renew its nominal provider
identity together with Leto's typed constructor after the target is accepted.
That renewal requires fresh graph/API/native/package qualification.
