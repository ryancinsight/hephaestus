//! Domain contracts: errors, typed device buffers, and accelerator seams.

/// Typed device-buffer contract.
pub mod buffer;
/// Shared CPU-side panel factorisation routines for blocked decomposition.
pub mod decomposition;
/// Compute-device acquisition and transfer seam.
pub mod device;
/// Kernel-dialect markers and per-dialect scalar tokens.
pub mod dialect;
/// Backend-neutral elementwise operations over strided n-D views.
pub mod elementwise;
/// Error contracts shared by all backends.
pub mod error;
/// Backend-neutral kernel authoring: interface and source declarations.
pub mod interface;
/// Kernel-dispatch contracts shared by accelerator backends.
pub mod kernel;
/// Launch-shape vocabulary for occupancy-planned dispatch.
pub mod launch;
/// Zero-sized operation markers with per-dialect shader expressions.
pub mod ops;
/// Shared narrowing/error helpers for dispatch planning.
pub(crate) mod planning;
/// Backend-neutral axis-reduction validation and dispatch planning.
pub mod reduction;
/// Backend-neutral axis-scan validation and dispatch planning.
pub mod scan;
/// Backend-neutral volume ray-integral geometry and validation.
/// Device-neutral sparse operator contracts.
pub mod sparse;
/// Backend-neutral two-dimensional Laplacian stencil parameters.
pub mod stencil;
/// Authored-kernel dispatch seam: prepared pipelines, bindings, streams.
pub mod stream;
/// Dense vector-operation contracts.
pub mod vector;
/// Device-neutral strided views over backend buffers.
pub mod view;
/// Volume contracts.
pub mod volume;
