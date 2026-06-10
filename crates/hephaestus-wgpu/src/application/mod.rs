//! Monomorphized compute dispatch over the wgpu device.

/// Elementwise binary kernels.
pub mod elementwise;
/// Reduction compute operations.
pub mod reduction;
/// Strided-layout-aware dispatch over leto layout metadata.
pub mod strided;
/// WGSL scalar-type mapping.
pub mod wgsl;
