# ADR 0060: Keep pooled buffer ownership inside the provider

- Status: Accepted
- Date: 2026-09-08
- Change class: major, architecture
- Board item: [HEPH-WGPU-POOL-OWNERSHIP](../../backlog.md#heph-wgpu-pool-ownership)

## Contract and evidence

Transient pools must receive only allocations created for that exact device
and pool role. Public getters return raw WGPU buffers, and public recycle
methods accept any raw buffer. The pools select solely by physical size;
WGPU resource identifiers are instance-local, so foreign handles can resolve
to unrelated local allocations before native validation reports a failure.
Required-device run `68d62fe9` reproduces two safe caller sequences: foreign
staging recycling makes typed readback panic on an aliased copy; foreign
uniform recycling makes a scalar kernel bind the owner's storage input as a
uniform and panic. No data-corruption claim is inferred from those panics.

A complete Rust source search across Atlas finds the acquisition/recycle and
guard APIs only inside this WGPU crate. Public prepared-operation types hold
guards in private fields; no public operation requires callers to name them.

## Decision

Make transient acquisition and its returned owning buffer crate-private.
One owner holds its raw allocation and captured origin: staging or uniform
on the creating device. The origin retains the exact pool through an `Arc`,
plus staging accounting where required; it does not clone adapter metadata
or unrelated device caches. Its destructor returns only to that origin. Remove
public raw recycle endpoints and both exported guard aliases; callers use
the returned owner directly instead of acquiring raw storage and separately
constructing a recycling guard. Borrowed raw access remains crate-private for
bindings and commands. Internal temporary fields/collections retain owners
until their existing submission or prepared-operation lifetime ends.

This closes the trust boundary by removing unneeded external pool access.
No public constructor, Deref, raw reference or conversion exposes a pooled
allocation. Keeping a public RAII owner with raw access is rejected: safe
callers could clone a raw handle, drop its owner to recycle it, then destroy
or mutate a later borrower's allocation. Comparing raw IDs cannot establish
ownership across WGPU instances. A runtime fallback that discards arbitrary
foreign inputs would retain the invalid insertion boundary.

The origin is a closed enum whose variants carry the creating pool. No
public operation varies by staging/uniform type after manual recycling is
removed, so a public generic marker would add no enforceable capability.
There is one destructor and no function-pointer recycling policy.

## Review

2026-09-08: independent source review accepts the private pool boundary and
captured pool/accounting ownership. The raw-handle escape alternative is
rejected above. Device execution and SemVer gates remain separate evidence.

## Migration

`get_staging_buffer`, `get_uniform_buffer`, `recycle_staging_buffer`, and
`recycle_uniform_buffer` cease to be public. `StagingBufferGuard` and
`UniformBufferGuard` exports are removed. High-level `ComputeDevice` transfers
and kernels are unchanged. Custom WGPU interop callers allocate and own their
buffers through `WgpuDevice::device()` and WGPU's explicit error scopes; those
allocations never enter provider pools. `clear_transient_pools` remains public
for hosts that explicitly release retained provider allocations. This is a
public breaking change; no compatibility shim or version/release bump is
included in this development increment.

## Verification and limits

Preserve the original foreign-recycling sequences as compile-fail public API
examples. Test real staging/uniform reuse on two independently acquired devices
with exact transfer/kernel values, alongside the full WGPU suite and bounded
readback tests. Strict Clippy, docs, SemVer comparison and independent design
and diff review cover the migration. The memory argument is ownership of
internal handles plus public inaccessibility; it does not claim general raw
WGPU queue safety, physical OOM recovery or measured performance improvement.
