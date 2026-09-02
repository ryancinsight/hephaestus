//! CUDA providers for generic pooling and sliding-window operations.

use hephaestus_core::{CFamilyPoolingOps, CFamilySlidingWindowOps};

use crate::CudaDevice;

/// CUDA's generic pooling provider.
pub type CudaPoolingOps = CFamilyPoolingOps<CudaDevice>;

/// CUDA's generic sliding-window provider.
pub type CudaSlidingWindowOps = CFamilySlidingWindowOps<CudaDevice>;
