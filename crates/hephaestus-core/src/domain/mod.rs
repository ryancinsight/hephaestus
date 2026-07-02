//! Domain contracts: errors, typed device buffers, and accelerator seams.

/// Typed device-buffer contract.
pub mod buffer;
/// Shared CPU-side panel factorisation routines for blocked decomposition.
pub mod decomposition;
/// Compute-device acquisition and transfer seam.
pub mod device;
/// Kernel-dialect markers and per-dialect scalar tokens.
pub mod dialect;
/// Error contracts shared by all backends.
pub mod error;
/// Kernel-dispatch contracts shared by accelerator backends.
pub mod kernel;
/// Launch-shape vocabulary for occupancy-planned dispatch.
pub mod launch;
/// Zero-sized operation markers with per-dialect shader expressions.
pub mod ops;
