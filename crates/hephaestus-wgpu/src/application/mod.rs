//! Monomorphized compute dispatch over the wgpu device.

/// Elementwise binary kernels.
pub mod elementwise;
/// Reduction compute operations.
pub mod reduction;
/// WGSL scalar-type mapping.
pub mod wgsl;
