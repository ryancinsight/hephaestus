use hephaestus_core::{
    DeviceBuffer, DispatchGrid, FixedFd3DParams, HephaestusError, MultiStorageKernel, Result,
};

use super::kernel::FIXED_FD_3D_KERNEL;
use crate::RocmDevice;
use crate::application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
use crate::infrastructure::RocmBuffer;

const WORKGROUP: [usize; 3] = [4, 4, 4];

/// Compiled HIP fixed-scheme sweep kernel, with its transpose.
#[derive(Debug)]
pub struct FixedFd3DKernel {
    sweep: RocmMultiStorageKernel,
    adjoint: RocmMultiStorageKernel,
}

impl FixedFd3DKernel {
    /// Compile the sweep and its transpose for a ROCm device.
    pub fn new(_device: &RocmDevice) -> Result<Self> {
        let sweep = RocmMultiStorageKernel::new(
            "hephaestus-fixed-fd-3d-sweep",
            FIXED_FD_3D_KERNEL,
            "fixed_fd_sweep",
            &[0, 1],
            [4, 4, 4],
            0,
        )?;
        let adjoint = RocmMultiStorageKernel::new(
            "hephaestus-fixed-fd-3d-adjoint",
            FIXED_FD_3D_KERNEL,
            "fixed_fd_adjoint",
            &[0, 1],
            [4, 4, 4],
            0,
        )?;
        Ok(Self { sweep, adjoint })
    }

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    pub fn sweep(
        &self,
        device: &RocmDevice,
        input: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_storage(input.len(), output.len())?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.output_dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        MultiStorageKernel::<RocmDevice, FixedFd3DParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.sweep,
            device,
            [
                RocmStorageBinding::new(0, input),
                RocmStorageBinding::new(1, output),
            ],
            params,
            grid,
        )
    }

    /// Sweep the transpose of the scheme in `params` along its axis,
    /// `upstream` into `grad`.
    pub fn adjoint(
        &self,
        device: &RocmDevice,
        upstream: &RocmBuffer<f32>,
        grad: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_adjoint_storage(upstream.len(), grad.len())?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        MultiStorageKernel::<RocmDevice, FixedFd3DParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.adjoint,
            device,
            [
                RocmStorageBinding::new(0, upstream),
                RocmStorageBinding::new(1, grad),
            ],
            params,
            grid,
        )
    }
}

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
