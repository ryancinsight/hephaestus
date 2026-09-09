//! WGPU implementation of the dense vector-operation seam.

mod kernels;
mod ops;

pub use ops::{WgpuPreparedDot, WgpuPreparedNorm, WgpuVectorOps};
