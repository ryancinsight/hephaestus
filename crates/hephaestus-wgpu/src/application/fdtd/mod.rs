//! Provider-owned three-dimensional acoustic FDTD kernels.

mod fdtd3d;

pub use fdtd3d::{Fdtd3dKernel, WgpuFdtd3dOps};
