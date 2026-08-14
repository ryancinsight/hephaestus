//! One generic accelerator layer over a narrow device-API seam.
//!
//! Backends that compile kernel source at runtime and launch it against raw
//! device addresses — CUDA through NVRTC, ROCm through hipRTC — differ only
//! along the device-API axis: the compiled-kernel handle, the device-pointer
//! representation, and the pipeline-cache key. Everything above that axis
//! (validation, cache-key construction, shared-memory sizing, launch
//! geometry, the allocating and non-allocating entry points, and the
//! convenience operators) is one algorithm.
//!
//! [`DeviceApi`] is that axis. Op families in this module are written once
//! against it and monomorphize per backend, so a vendor crate contains only
//! its device-API implementation and never a copy of the orchestration.
//!
//! This is the same consolidation
//! [`BlockedDecompositionBackend`](crate::BlockedDecompositionBackend)
//! performed for blocked LU, generalized to the launch layer and — unlike
//! that trait — parameterized over the scalar rather than fixed to `f32`.

/// The device-API seam: compiled kernels, device pointers, and launches.
pub mod device_api;
/// Generic rank-2 axis prefix/suffix scan over the device-API seam.
pub mod scan;
