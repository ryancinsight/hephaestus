# Leto 0.47 compatibility for borrowed CSR consumers

Base: `672c5abc88f3984e2b7ece439b3dd4ed5f1f7f68`. Target: 0.22.1. Classification: [patch].
This isolated export leaves every existing checkout and its owner untouched.

## Contract

Leto draft #362 implements accepted ADR 0036. Its 0.47 addition preserves the
existing default-index/ordering and scalar/SIMD APIs; its minor API comparison
against 0.46 reports no removal. This provider uses the existing layout, array
and operation vocabulary. No new CSR API appears in its public interface.

The old `^0.46.0` requirement rejects the source graph when Coeus selects 0.47.
Accept `>=0.46.0, <0.48.0` for both Leto layers. The interval retains the
current 0.46 cohort and admits the verified additive 0.47 line; later pre-1.0
minor lines require a new compatibility audit. Every first-party source stays
canonical, with one resolved provider identity. No conversion adapter, path
patch, manifest revision pin, backend feature or Rust source changes.

Published packages whose dependency closure changes receive patch versions.
The committed lock retains merged provider defaults (currently Leto 0.46).
A separate review consumer must exercise Leto 0.47 and both matching provider
bindings before Coeus claims its complete graph. This source compatibility
change does not authorize provider merge, registry publication or Pages.

## Executed source evidence

Runtime source: `a153f475fbee1042fc323a845400da89c9737c94`. All Rust implementation/test files and
all external locked packages remain byte-identical to the recorded current
base. Manifest resolution and `metadata --offline --locked` pass without local
path/patch overrides or manifest revision pins.

All actual hosted candidate checks pass. The exact-source API/dependency/
version/lock/ADR gates pass; configured release/informational jobs remain
separate from a release authorization. CI evidence:
[run 38781083047](https://github.com/ryancinsight/hephaestus/actions/runs/38781083047).
Host verification, real macOS Metal, WGPU software-adapter and adapterless
CUDA/ROCm gates pass. NVIDIA/AMD hardware jobs do not execute; no such hardware
result or registry publication is claimed. A fresh unseeded review-source
consumer resolves seven canonical providers with zero excess identities.

The actual Coeus 0.47 review graph passes 1272 native cases, strict workspace
Clippy, 159 doctests and thirteen strict/isolation memory cases in each borrow
model. These establish the real matching provider boundary, independently of
the committed 0.46 graph. Coeus's new hosted memory job is being rechecked after
correcting its interpreter installer. Its unseeded defaults still require the
provider reviews to integrate first.

No main merge, tag, registry publication or Pages deployment occurs in this
compatibility review. Those external outcomes require their own actual checks
and authorization. This closure changes records only; runtime source above
remains the reproducible implementation reference.
