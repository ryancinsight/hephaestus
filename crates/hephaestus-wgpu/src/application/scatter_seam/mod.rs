//! WGPU implementation of [`hephaestus_core::ScatterOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell first copies its input cell, then scans every
//! source cell in ascending row-major order and adds the ones whose index
//! maps to it. The scan is output-parallel because repeated indices would
//! race an input-parallel scatter (and WGSL has no floating-point
//! atomics); the ascending scan keeps every backend bitwise-identical on
//! floating-point sums, exactly as the embedding scatter-add does. `+`
//! applies to every [`DialectScalar<Wgsl>`](hephaestus_core::DialectScalar)
//! type — numeric scalars and numeric vectors alike — so the kernel needs
//! no further scalar bound.
//!
//! The copy and the scan are two launches, not one: the scan's four
//! storage buffers (indices, source, output, error flag) already fill the
//! portable `max_storage_buffers_per_shader_stage` limit of four, so the
//! input cannot join the same bind group. The copy kernel runs first and
//! the scan seeds its accumulator from the copied output cell — each
//! thread still touches only its own cell, so no cross-thread sharing
//! appears.
//!
//! A source cell maps to its thread's cell exactly when the output offset
//! recomputed from its coordinates equals the thread's own offset, which
//! is a sound comparison only because validation proves the output layout
//! injective. The index *values* are device-resident data, so an
//! out-of-range value cannot be caught before dispatch the way a
//! host-known shape mismatch can. The kernel detects it and reports it
//! through a one-`u32` atomic error flag, downloaded after the dispatch
//! completes, rather than clamping or wrapping (error-handling restraint:
//! a data-dependent fault is surfaced, never silently masked). A faulting
//! update maps to no cell and is skipped; the output is left partially
//! accumulated, which is irrelevant once the call returns an error.
//!
//! When the index holds no updates, a smaller copy kernel runs instead of
//! the scan: binding the empty index/source buffers would violate the
//! nonzero minimum binding size, and there is nothing to accumulate.

mod dispatch;
mod shader;

pub use dispatch::{WgpuScatterOps, scatter_add_into};
