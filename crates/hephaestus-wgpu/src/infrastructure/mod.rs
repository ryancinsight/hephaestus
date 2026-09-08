//! wgpu device acquisition and typed buffer storage.

/// Typed device buffers over `wgpu::Buffer`.
pub mod buffer;
/// Device/queue acquisition and the `ComputeDevice` implementation.
pub mod device;
pub(crate) mod pool;
