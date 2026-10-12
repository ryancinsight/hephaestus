use hephaestus_core::{FixedFd3DParams, Result};

use super::kernel::FixedFd3DKernel;
use crate::RocmDevice;
use crate::infrastructure::RocmBuffer;

/// Provider-owned implementation of [`FixedFd3DOps`](hephaestus_core::FixedFd3DOps) for Rocm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmFixedFd3DOps;

impl hephaestus_core::FixedFd3DOps<RocmDevice> for RocmFixedFd3DOps {
    type FixedFd3D = FixedFd3DKernel;

    fn prepare_fixed_fd_3d(&self, device: &RocmDevice) -> Result<Self::FixedFd3D> {
        FixedFd3DKernel::new(device)
    }

    fn fixed_fd_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::FixedFd3D,
        input: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.sweep(device, input, output, params)
    }

    fn fixed_fd_adjoint_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::FixedFd3D,
        upstream: &RocmBuffer<f32>,
        grad: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.adjoint(device, upstream, grad, params)
    }
}
