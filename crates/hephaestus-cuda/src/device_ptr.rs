//! The device-address type shared by the CUDA buffer unit and its consumers.

/// A raw CUDA device pointer (`CUdeviceptr`), an opaque device address.
///
/// Kept as the CUDA driver's ABI type without exposing the CUDA driver in
/// public APIs; consumers see an opaque integer address for custom kernel
/// launches. One definition serves the `cuda` backend and the unsafe-free
/// stub, so no consumer needs a feature-gated fallback alias.
pub type DevicePtr = u64;
