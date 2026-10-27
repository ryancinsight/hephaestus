//! WGPU implementation of [`hephaestus_core::IndexPutOps`].
//!
//! One packed-metadata layout serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK),
//! with two update kernels behind the `accumulate` flag. Both run after a
//! copy kernel first writes the base input into the output:
//!
//! * Replace mode is input-parallel: one thread per values cell reads the
//!   index at its own `dim` coordinate, substitutes it into the `dim` lane,
//!   and overwrites the addressed output cell. Duplicate indices race by
//!   design — `torch.index_put` leaves duplicate replace order undefined,
//!   so no clause pins it.
//! * Accumulate mode is output-parallel: one thread per output cell seeds
//!   its accumulator from the copied base cell, then scans every selection
//!   in ascending order and adds the values slices whose index maps to it.
//!   The scan is output-parallel because repeated indices would race an
//!   input-parallel scatter (and WGSL has no floating-point atomics); the
//!   ascending scan keeps every backend bitwise-identical on
//!   floating-point sums, exactly as the scatter seam does. `+` applies to
//!   every [`DialectScalar<Wgsl>`](hephaestus_core::DialectScalar) type —
//!   numeric scalars and numeric vectors alike — so the kernel needs no
//!   further scalar bound.
//!
//! The copy and the update are two launches, not one: the update's four
//! storage buffers (indices, values, output, error flag) already fill the
//! portable `max_storage_buffers_per_shader_stage` limit of four, so the
//! input cannot join the same bind group. Each thread still touches only
//! its own cell, so no cross-thread sharing appears.
//!
//! The index *values* are device-resident data, so an out-of-range value
//! cannot be caught before dispatch the way a host-known shape mismatch
//! can. The kernel detects it and reports it through a one-`u32` atomic
//! error flag, downloaded after the dispatch completes, rather than
//! clamping or wrapping (error-handling restraint: a data-dependent fault
//! is surfaced, never silently masked). A faulting update is skipped; the
//! output is left partially written, which is irrelevant once the call
//! returns an error.
//!
//! When the index holds no updates, a smaller copy kernel runs instead of
//! the update: binding the empty index/values buffers would violate the
//! nonzero minimum binding size, and there is nothing to write.

mod dispatch;
mod shader;

pub use dispatch::{WgpuIndexPutOps, index_put_into};
