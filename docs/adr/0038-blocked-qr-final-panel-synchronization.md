# ADR 0038: Finish the blocked-QR tail after one readback

- Status: accepted
- Date: 2026-07-28
- Scope: backend-neutral packed QR panels and the WGPU blocked-QR schedule
- Change class: `[minor]`

## Context

The WGPU blocked-QR path factors each panel on the CPU and applies that panel's
Householder reflectors to trailing columns on the GPU. Every panel therefore
downloads its active matrix region and waits for mapping before CPU
factorization. At the retained 70×35 benchmark shape, two 32-column panels
produce two device-to-host synchronization boundaries. Component profiling
measures the synchronization floor at 213–231 µs while CPU panel
factorization costs 26–28 µs, so synchronization count is the binding cost.

The path already owns two reusable device buffers of `m * block_size` elements:
`temp_compact_buf` for a panel and `vectors_dev` for packed reflectors. The last
tail is at most one block wide, so those two buffers can hold the penultimate
panel and final tail simultaneously without changing the scratch bound.

## Decision

Add one backend-neutral operation next to `panel_qr_packed` that applies a
factored packed panel's Householder reflectors to a compact row-major trailing
matrix. It uses the same reflector order and native `f32` arithmetic as the
existing WGPU kernel.

Add paired matrix-region download and write operations that encode two gathers
or scatters into one command encoder and submit once. The single-region
operations and paired operations share their validation, binding, encoding,
mapping, and host-copy implementation.

When a blocked QR iteration has a non-empty tail no wider than one block, gather
the active panel and tail together, factor the panel, apply its reflectors to the
tail on the CPU, factor the tail, and write both final regions together. This
removes the final panel readback and the penultimate panel's trailing GPU
dispatch. Earlier panels retain the GPU trailing update. `panel`,
`packed_vectors`, `temp_compact_buf`, and `vectors_dev` are reused, so neither
host algorithm scratch nor persistent device scratch capacity increases.

Both compact results must remain independently mappable until the shared poll
completes. The paired readback therefore holds a second transient staging
buffer of `m * tail_cols * size_of::<f32>()` bytes, capped at one block, for the
duration of that readback. The buffer returns to the existing pool immediately
after mapping. This bounded peak-staging increase is the explicit memory cost
of removing one synchronization; the benchmark and artifacts report it beside
the latency result.

## Alternatives rejected

- Download the final tail in the next loop iteration: rejected because it keeps
  the measured synchronization boundary.
- Allocate a wider `m * (block_size + tail)` compact buffer: rejected because it
  trades synchronization latency for a larger peak scratch allocation.
- Duplicate the reflector application inside the WGPU QR module: rejected
  because CUDA and future hybrid providers need one backend-neutral packed-panel
  contract.
- Finish every trailing block on the CPU: rejected because wide trailing work
  has enough arithmetic intensity to remain on the GPU; only the final
  at-most-one-block tail crosses this boundary.
- Change the block width: rejected because it changes both arithmetic scheduling
  and the benchmark scenario without addressing the synchronization root cause.

## Verification

The packed-panel operation has analytical and differential tests for reflector
order, empty tails, malformed storage, and a direct full-panel QR comparison.
WGPU contracts cover 32/33/35/64/65-column boundaries and compare complete
`R` factors and least-squares solutions with Leto. A matched Criterion
before/after run uses the unchanged 70×35 workload and waits for device
completion. The benchmark reports latency only; value tests establish numerical
correctness. Source and allocation audits establish the unchanged persistent
scratch capacity and quantify the bounded transient staging increase. Exact-head
WGPU, CUDA, ROCm, and macOS Metal CI remain required
because the new core operation is shared provider surface.

## Revisit trigger

Revisit if a controlled benchmark shows the CPU tail work offsets the removed
synchronization, if the block width changes, or if a native device-resident QR
panel kernel eliminates the host boundary entirely.
