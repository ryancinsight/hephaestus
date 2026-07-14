# ADR 0006: WGPU 30 provider ABI

- Status: Accepted
- Date: 2026-07-13
- Change class: major

## Context

Hephaestus 0.12 publicly re-exports WGPU and exposes WGPU buffers, devices,
queues, descriptors, and poll types through its provider contract. Apollo must
therefore use the same WGPU major. WGPU 30.0.0 is the current registry release
and requires Rust 1.87. Apollo and the local Atlas toolchain satisfy that MSRV.
WGPU 26 also retains the archived `paste` advisory through its Metal backend and
caps `ordered-float`, so keeping 26 leaves known supply-chain residue.

WGPU HAL 30 and `gpu-allocator` 0.28 expose a resolver-sensitive Windows edge:
the allocator accepts `windows` 0.53 through 0.62, while HAL exchanges D3D12
types from 0.62. A graph already containing Moirai PAL's 0.58 binding can cause
Cargo to reuse 0.58 for the allocator and produce two incompatible D3D12 type
families inside HAL itself.

## Decision

Hephaestus owns the migration. Update its single workspace WGPU dependency to
30.0.0 and rewrite every affected provider call site directly against WGPU 30;
do not add aliases, adapters, or dual-version branches. Preserve Hephaestus's
backend-neutral core contracts. Bump the pre-1.0 workspace version to 0.13.0
because public re-exported WGPU types change identity. After the provider gates
pass and the commit is pushed, Apollo pins that exact revision, updates its own
WGPU SSOT to 30.0.0, removes the obsolete advisory exception, and repeats its
release gates.

WGPU 30 also makes the former Mnemosyne staging callback unrepresentable:
`BufferViewMut` is an explicit write-only byte view because mapped memory may
be write-combining, while `MemoryBackend` promises a generally writable raw
pointer. Mnemosyne 0.4.0 deletes that backend. Hephaestus deletes the callback
registry and pointer ownership, makes `WgpuDevice::new` infallible, retains its
provider-owned transient transfer pools, and rejects concrete memory-tier
requests WGPU cannot guarantee.

Until the allocator narrows its Windows requirement, the Windows WGPU package
declares an exact 0.62.2 dependency. This constrains the allocator/HAL edge to
one type identity while leaving Moirai PAL's independent 0.58 API intact. The
pin is dependency-resolution policy, not a runtime adapter or compatibility
surface; remove it when upstream makes the constraint redundant.

## Rejected alternatives

- Upgrade Apollo alone: creates incompatible WGPU type families across the
  public provider boundary.
- Retain WGPU 26 beside 30: preserves the ABI split and duplicate GPU runtime.
- Hide changes behind compatibility aliases: retains obsolete type identity and
  violates the direct-migration contract.
- Remove the public WGPU re-export in this increment: that is a separate
  provider-boundary redesign with a larger consumer migration surface.

## Failure modes

- Descriptor-field and polling API changes fail at compile time.
- Two WGPU majors in the consumer graph fail the dependency-source audit.
- Driver-visible behavior changes fail existing value-semantic WGPU nextest
  contracts under the committed serial GPU test group.
- The advisory is not closed unless `cargo audit`, `cargo deny check`, and
  inverse dependency inspection all confirm the old graph is absent.

## Verification

Run format, warning-denied Clippy for all WGPU-consuming targets, nextest under
the committed timeout, doctests, warning-denied rustdoc, dependency-tree source
inspection, RustSec, cargo-deny, and semver classification. Apollo repeats the
same release gate plus its Python boundary tests after repinning.
