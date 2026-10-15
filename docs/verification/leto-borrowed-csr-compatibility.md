# Leto 0.47 compatibility for borrowed CSR consumers

Base: `b03e91761cc929d046bd0d911bd573c699560d8e`. Target: 0.22.2. Classification: [patch].
This isolated export leaves every existing checkout and its owner untouched.

## Contract

Leto #362 implements accepted ADR 0036. Its additive 0.47 API preserves the
existing default-index/ordering and scalar/SIMD APIs. This provider uses the
existing layout, array and operation vocabulary; no CSR API enters its public
interface. Accept `>=0.46.0, <0.48.0` for both Leto layers, retaining the merged
0.46 cohort and admitting the verified 0.47 line. Later pre-1.0 minor lines
require a new compatibility audit.

Every first-party source remains canonical, with one resolved provider
identity. No conversion adapter, path patch, manifest revision pin, backend
feature or Rust source changes. The committed lock follows merged defaults
(currently Leto 0.46). Independent Coeus review #533 exercises the actual 0.47
source graph before claiming that integration.

## Current-base preservation and version ownership

The first candidate `a153f475` passes host verification, real Metal, WGPU
software-adapter, adapterless CUDA/ROCm and actual locked-source API gates in
[run 38781083047](https://github.com/ryancinsight/hephaestus/actions/runs/38781083047).
Its unseeded review-source consumer resolves seven canonical providers without
an excess identity. Those results apply to that recorded runtime revision.

Concurrent main changes #496/#498 publish 0.22.1 while this review is open.
The version guard rejects reusing that published version for different source.
This candidate therefore preserves the complete current main, including both
CUDA reduction fixes, and advances the compatibility patch to 0.22.2.
All 1337 current Rust files and all 228 external locked records remain
byte-identical to the new base. Only compatibility/version metadata and
review documentation differ. Runtime `e867be0d346064ea5f956a8f78f81c31bff88af4` passes every actual hosted
gate: host native/strict/docs, real Metal, WGPU software-adapter, CUDA/ROCm
feature and adapterless contracts, all six locked-source API comparisons,
and version/dependency/lock/ADR guards. The portable API/host record is
[run 38792609338](https://github.com/ryancinsight/hephaestus/actions/runs/38792609338).
NVIDIA/AMD hardware and configured release/informational jobs skip. A new
unseeded 0.22.2 review-source consumer resolves seven providers with no excess
identity. No registry/default availability claim follows from those results.

The actual Coeus 0.47 integration has independent native, strict, doctest,
exact-value/performance and strict-isolation memory evidence in its review.
Coeus's complete current-main review lock selects this exact source and
passes 1272 native cases, strict all-workspace/all-target Clippy, 159 doctests
and warning-denied documentation. All 231 registry records match Coeus main.
The provider's source and the downstream CPU kernel have explicit revision
receipts; later documentation-only commits preserve their bytes. Default/registry availability remains a
provider-first integration gate.

No main merge, tag, registry publication or Pages deployment occurs in this
compatibility review. NVIDIA/AMD hardware execution is separate from
adapterless hosted checks. External outcomes require their own actual checks
and authorization.
