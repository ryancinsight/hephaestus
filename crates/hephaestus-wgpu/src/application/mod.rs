//! Monomorphized compute dispatch over the wgpu device.

/// Elementwise binary kernels.
pub mod elementwise;
/// Linear algebra compute operations.
pub mod linalg;
pub(crate) mod pipeline;
/// Reduction compute operations.
pub mod reduction;
/// Prefix and suffix scan compute operations.
pub mod scan;
/// Strided-layout-aware dispatch over leto layout metadata.
pub mod strided;
/// WGSL scalar-type mapping.
pub mod wgsl;
