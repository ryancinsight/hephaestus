//! Domain contracts: errors, typed device buffers, and the compute-device seam.

/// Typed device-buffer contract.
pub mod buffer;
/// Shared CPU-side panel factorisation routines for blocked decomposition.
pub mod decomposition;
/// Compute-device acquisition and transfer seam.
pub mod device;
/// Error contracts shared by all backends.
pub mod error;
/// Launch-shape vocabulary for occupancy-planned dispatch.
pub mod launch;
