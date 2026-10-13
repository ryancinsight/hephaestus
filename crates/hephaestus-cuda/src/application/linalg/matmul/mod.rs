//! Matrix multiplication on the CUDA device.
//!
//! A bespoke 16×16 shared-memory tiled kernel, authored as CUDA C generic over
//! the scalar (`T::TYPE_TOKEN` substitutes the device type token) and launched
//! directly through `cuLaunchKernel`. [`batched_matmul`] dispatches the batch
//! dimension via `blockIdx.z` in a dedicated kernel (CU-P5) rather than
//! looping [`matmul_into`] once per batch element — each loop iteration was a
//! separate `cuLaunchKernel` plus (on Windows) a `cuCtxSynchronize` context
//! drain per `launch_kernel` call (KS-8), serializing what should be one
//! dispatch. `MAX_GRID_Z` chunks batches beyond CUDA's hardware grid.z limit
//! (65535 on every current compute capability) into multiple launches.

mod batched;
mod shaders;
mod single;

pub use batched::{batched_matmul, batched_matmul_into};
pub use single::{matmul, matmul_into};

#[cfg(all(test, feature = "cuda"))]
pub(crate) use shaders::{batched_matmul_shader_source, matmul_shader_source};

#[cfg(all(test, feature = "cuda"))]
#[path = "tests.rs"]
mod tests;
