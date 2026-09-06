//! Provider-owned stencil operators for finite-difference PDEs.
//!
//! These kernels live in Hephaestus so that consumers (`cfd-core`, etc.) remain
//! thin typed callers rather than owning WGSL source and dispatch details.

mod laplacian2d;
mod staggered3d;

pub use laplacian2d::WgpuStencilOps;
pub use laplacian2d::{BoundaryCondition, Laplacian2DKernel, Laplacian2DParams, LaplacianPolarity};
pub use staggered3d::WgpuStaggered3DOps;
pub use staggered3d::{Staggered3DKernel, Staggered3DParams, StaggeredAxis};
