use hephaestus_core::{FixedFd3DParams, Result};

use super::kernel::FixedFd3DKernel;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Provider-owned implementation of [`hephaestus_core::FixedFd3DOps`] for
/// WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFixedFd3DOps;

impl hephaestus_core::FixedFd3DOps<WgpuDevice> for WgpuFixedFd3DOps {
    type FixedFd3D = FixedFd3DKernel;

    fn prepare_fixed_fd_3d(&self, device: &WgpuDevice) -> Result<Self::FixedFd3D> {
        FixedFd3DKernel::new(device)
    }

    fn fixed_fd_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::FixedFd3D,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.sweep(device, input, output, params)
    }

    fn fixed_fd_adjoint_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::FixedFd3D,
        upstream: &WgpuBuffer<f32>,
        grad: &WgpuBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.adjoint(device, upstream, grad, params)
    }
}
