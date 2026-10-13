use hephaestus_core::{FixedFd3DParams, Result};

use crate::CudaDevice;
use crate::infrastructure::buffer::CudaBuffer;

use super::kernel::FixedFd3DKernel;

/// Provider-owned implementation of [`hephaestus_core::FixedFd3DOps`] for
/// CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaFixedFd3DOps;

impl hephaestus_core::FixedFd3DOps<CudaDevice> for CudaFixedFd3DOps {
    type FixedFd3D = FixedFd3DKernel;

    fn prepare_fixed_fd_3d(&self, device: &CudaDevice) -> Result<Self::FixedFd3D> {
        FixedFd3DKernel::new(device)
    }

    fn fixed_fd_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::FixedFd3D,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.sweep(device, input, output, params)
    }

    fn fixed_fd_adjoint_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::FixedFd3D,
        upstream: &CudaBuffer<f32>,
        grad: &CudaBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.adjoint(device, upstream, grad, params)
    }
}
