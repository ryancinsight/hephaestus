//! Provider-owned stencil operators for finite-difference PDEs.
//!
//! These kernels live in Hephaestus so that consumers (`cfd-core`, etc.) remain
//! thin typed callers rather than owning WGSL source and dispatch details.

mod laplacian2d;

pub use laplacian2d::{BoundaryCondition, Laplacian2DKernel, Laplacian2DParams};
