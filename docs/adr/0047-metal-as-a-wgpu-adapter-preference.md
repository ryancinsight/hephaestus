# ADR 0047 — Metal is a WGPU adapter preference, not a backend crate

- Status: Accepted
- Date: 2026-08-03
- Refs: atlas `backlog.md#atlas-arch-009` (the mandating item); the
  [ComputeBackend conformance triage](../audit/2026-07-28-computebackend-conformance-triage.md)
  §"topology" (raised the evidence, deliberately did not decide); ADR 0041
  (conformance crate, whose suite this changes the instantiation set of);
  ADR 0012 (the ROCm backend, the comparison case for what a vendor crate is).

## Context

`hephaestus-metal` presents Metal as a peer backend alongside
`hephaestus-cuda`, `hephaestus-rocm`, and `hephaestus-wgpu`. It is 5 449
source lines plus a 2 606-line test suite.

It contains **no Metal device API**. There is no `metal::`, no `objc`, no
`MTLDevice`, and no MSL shader anywhere in the crate. `MetalDevice` is a
newtype over `WgpuDevice`:

```rust
pub struct MetalDevice { pub(crate) inner: WgpuDevice }

pub fn try_default() -> Result<Self> {
    let inner = WgpuDevice::try_metal("hephaestus-metal-device")?;
    Ok(Self { inner })
}
```

Every `application/*` module forwards to `hephaestus-wgpu`
(`reduction.rs` 521 lines / 24 forwards, `decomposition.rs` 268/23,
`sparse.rs` 252/23, `linalg.rs` 246/18). Its only unique public surface is
the two escape hatches `wgpu_device` and `wgpu_buffer` — which
consumers already reach through, so the crate does not in fact insulate
anyone from the WGPU underneath.

The comparison against its siblings is what settles the question. A
per-vendor crate exists to own a device API; the accelerator layer above
it is written once and monomorphized:

| crate | src lines | native device-API refs | depends on wgpu |
| --- | --- | --- | --- |
| `hephaestus-cuda` | 18 600 | 66 | no |
| `hephaestus-rocm` | 17 925 | 62 | no |
| `hephaestus-metal` | 5 449 | **0** | **yes** |

CUDA and ROCm earn their crates: each owns a distinct device API that
nothing else in the stack can express. Metal owns nothing. It is the
inverse of the sanctioned shape — a per-vendor crate carrying a
duplicated accelerator layer and *no* device-API impl.

Decisively, `hephaestus-wgpu` already owns Metal selection as a
first-class device-preference path. It ships a whole family of
constructors for it — `try_metal`, `try_metal_with_device_preference_and_optional_device_features_and_limits`,
`try_enumerate_metal_with_optional_device_features_and_limits` — the last
of which filters adapters on `matches!(info.backend, wgpu::Backend::Metal)`.
Metal targeting is therefore not at stake in this decision; it already
exists one layer down and is where the real work lives.

## Decision

**Retire `hephaestus-metal` as a crate.** Metal is an adapter preference
of the WGPU backend, which is what `WgpuDevice::try_metal` already is.

1. **Delete the crate**, its workspace member entry, and its forwarding
   layer. No capability is lost: every operation it exposes is a forward
   to `hephaestus-wgpu`, which consumers can call directly.
2. **Leave `backend_name()` alone; the vendor identity is already
   exposed.** `MetalDevice::backend_name()` returns `"metal"` where
   `WgpuDevice` returns `"wgpu"`, and that is the wrapper's only
   observable difference. It needs no replacement: `WgpuDevice` already
   stores `adapter_info` and exposes it publicly, so a caller that
   genuinely needs the vendor reads
   `device.adapter_info().map(|i| i.backend) == Some(wgpu::Backend::Metal)`
   today, with no new API.

   `backend_name` deliberately keeps naming the *backend implementation*
   ("cuda", "rocm", "wgpu"), not the adapter vendor — they are different
   questions, and collapsing them would break the two CFDrs assertions on
   `backend_name() == "wgpu"`
   (`crates/cfd-core/tests/gpu_integration_test.rs:24`,
   `tests/gpu_integration.rs:17`, both holding a hephaestus `WgpuDevice`
   directly) for no gain. Retiring the crate simply retires `"metal"` as
   a backend name, which is correct: it was never a backend.
3. **Keep the `metal` feature on the `hephaestus` facade**, re-pointed:
   it now means "acquire a Metal-preferring `WgpuDevice`" rather than
   "compile a second copy of the operation surface". The consumer-facing
   spelling of intent survives the crate that used to carry it.
4. **The conformance suite loses an instantiation, not coverage.** The
   Metal instantiation runs the same clauses as the WGPU one against the
   same code path, so it asserts nothing WGPU does not already assert.
   Metal-adapter coverage is a matter of *which adapter CI acquires*, not
   of which crate the suite names — and that is honest about what was
   ever being verified.

### Alternatives rejected

- **Keep the crate as a stable vendor name (status quo).** Rejected: the
  insulation is not real. Consumers reach the `WgpuDevice` through the
  escape hatches, so the abstraction leaks by design, and the stack pays
  ~8 000 lines and a duplicated conformance instantiation for a name.
- **Keep `MetalDevice` as a type but delete the forwarding layer.** This
  is unstable: a typed device must implement the seam traits to be usable,
  and implementing them for a newtype over `WgpuDevice` *is* the
  forwarding layer. The option collapses back into the status quo the
  moment it is implemented.
- **Give Metal a native backend (`metal-rs`/`objc`).** Not rejected on
  merit — it is simply a different, much larger decision, and nothing in
  the current tree is waiting on it. Should a native Metal backend ever
  be written, it earns a crate the same way CUDA and ROCm do: by owning a
  device API. This ADR does not foreclose that; it declines to reserve a
  crate-shaped placeholder for it.

## Consequences

- **Breaking, so the item reclassifies `[arch]` → `[arch] [major]`.**
  `hephaestus-metal` is a published crate at 0.18.0; removing it is a
  major-version change for the workspace, with the migration path being
  the one-line substitution of `WgpuDevice::try_metal` for
  `MetalDevice::try_default`.
- **Downstream `coeus-metal` is the same shape one layer out**, and is
  the reason this decision is worth more than its own line count: the
  vendor dimension had been re-forked in a consumer. It is 1 233 lines
  with **no native Metal code and zero in-repo dependents** — a
  file-for-file copy of `coeus-rocm` (after normalizing the vendor token,
  the per-file diffs are 0, 0, 0, 2, 17, and 25 lines). `coeus-hephaestus`
  already implements the entire op surface generically for
  `HephaestusBackend<P>`, so the only content that is not mechanically
  reproducible from a ~56-line provider marker is one `fill_zero`
  override. Its disposition follows this ADR and is tracked under the
  Coeus per-vendor collapse (atlas `backlog.md#atlas-substrate-002`), not
  here — but the sizing above is now on record so that item starts from
  evidence rather than a re-survey.
- Expected failure mode: none in the observable contract, given decision
  point 2. The realistic risk is mechanical — a missed
  `hephaestus_metal::` reference or a stale workspace member entry — which
  the verification plan's stack-wide grep and `--all-targets` build catch.
  The `metal` **feature** name surviving on the facade is deliberate: it
  keeps consumers' spelling of intent working across the removal.

## Verification plan

1. `cargo check`/`nextest` green for the `hephaestus` workspace with the
   crate and its member entry removed, at `--all-targets`, plus the
   `metal` feature seam building in its re-pointed form.
2. The conformance suite passes with the Metal instantiation removed and
   no clause left unreferenced.
3. A stack-wide grep confirms no remaining `hephaestus_metal` reference
   outside Coeus's own tracked item, and no `backend_name` equality
   comparison depends on the old constant.
