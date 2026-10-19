# Fresh current-provider review-source resolution [patch]

Continue owner497 `5fe6ad9e5122efcd2aff9095279ed83c6720fa78`, preserving its
Moirai0.10 update and every Rust/backend byte. Its lock selects unmerged Leto362
objects through a default Git source, but fresh resolution can discover only
merged Leto0.46, whose Moirai^0.9.3 excludes current0.10. Name existing Leto362's
normal `codex/sparse-forward-contracts` branch for both layers, retaining the
owner's bounded>=0.46,<0.48 compatibility. No path/revision override, old pin or
policy exception is used. Remove the review branch after that cohort lands,
then regenerate the lock and all relevant gates. No owner ref is overwritten.

A fresh unseeded workspace lock succeeds and selects current Leto0.47 owner
b44b9f07b0fefab0f576ba971051514aca1c8ad6 and Moirai0.10. The final fresh lock also advances compatible registry records; the prior
owner lock is preserved as evidence and no version is downgraded. On this
actual final graph all439 selected host tests pass with zero skips; strict
workspace all-target check/Clippy, doctests, warning-denied docs and formatting
pass. Actual Rust1.95 required-device Metal gates on this fresh lock pass all
six feature/facade checks, both strict Clippy modes,96 minimal WGPU and184
default WGPU/Metal tests with zero skips, plus device doctests/Rustdoc. Both
required-device flags remain1. The final owner movement adds CUDA-only feature
imports and records; every tested Metal dependency/runtime/policy byte is
identical. Those owner fixes are adopted and the host checks renew afterward.
The owner's independent portable API receipts stay revision-scoped; final API,
Linux/other-device and hosted gates remain required.
Linux ROCm's required SDK is not bypassed on this Mac.

This proposal changes only two source selectors, fresh lock and records; target
0.22.2 and backend features stay. Obtain narrow exact-head review before a
remote continuation. Coordinate the same source selectors in Athena119 after
this provider step; unseeded default availability remains pending. No merge,
package publication, service action or credential change occurs.

## Pending Aequitas 0.4 boundary [minor]

[Aequitas141](https://github.com/ryancinsight/aequitas/pull/141) proposes law0.4.0
and binding0.2.0; implementation8c50883a01516aab8a22323d817cb8f27f59b307
is independently reviewed separately. The user approves ADR0020 and the normal version/common NumPy-PyO3.30 plan
on2026-10-19. Owner140 source coordination and actual hosted/free-threaded/book
gates remain pending. Current0.3 is
the verified public cohort for this checkpoint, not a permanent older-version
workaround. Do not consume an unaccepted candidate or edit its provider sources.

Static inspection of Harmonia, Athena119, Leto362 and Hephaestus497 finds no
generic `Quantity::powi<P>` or `TaggedDimension` producer requiring bound
propagation. Harmonia's `FieldEnvelope<T,D,RANK>` borrows quantities and does
not perform dimension powers or emit runtime tags. Leto and Hephaestus stencil
contracts extract concrete Length/Meter values, then use scalar `powi(i32)`;
that operation does not gain the proposed type-level bounds. Athena contains
neither surface. This is scoped source inspection, not a future-provider build.

After acceptance, renew all direct and transitive Aequitas requirements together
(including Horae and Leto/Hephaestus typed-stencil providers), regenerate fresh
locks and repeat typed quantity/time/spacing consumer and API gates against the
accepted exact target. Generic quantity-power producers, if introduced, need
`P: Integer + IsGreaterOrEqual<N2147483648,Output=B1> +
IsLessOrEqual<P2147483647,Output=B1>`; generic dimension-tag producers need each
axis in[-128,127]. Existing `D: TaggedDimension` consumers keep that bound.

Original compatible-provider receipt:

- Evidence: `e867be0d` passes every actual hosted native/strict/API/device-capable gate and fresh review-source resolution; Coeus consumes the exact 0.22.2 source in its passing native/strict/doc graph. Both concurrent CUDA fixes remain intact.

Original cohort audit record:

- Version ownership: concurrent main publishes 0.22.1, so this compatibility patch targets 0.22.2 and preserves current CUDA fixes. All actual source/hosted checks and the downstream native/strict/doc graph pass at `e867be0d`; default/registry availability remains provider-first. [The record](docs/verification/leto-borrowed-csr-compatibility.md) separates prior/current revision claims.

Original cohort audit record:

- Integration: master #494 (`b91180aa`) advances the same numeric providers; retain it as a parent and refresh all eight Git identities to current defaults, including merged Moirai `44a962f0`, Leto `8b4ab407` and Hermes `ac60146a`. All 204 registry records and every Rust source match that base.
