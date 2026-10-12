use hephaestus_core::{
    DeviceBuffer, DispatchGrid, FixedFd3DParams, HephaestusError, MultiStorageKernel, Result,
};

use super::source::{FIXED_FD_3D_KERNEL, WORKGROUP};
use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

/// Compiled CUDA fixed-scheme sweep kernel, with its transpose.
#[derive(Debug)]
pub struct FixedFd3DKernel {
    sweep: CudaMultiStorageKernel,
    adjoint: CudaMultiStorageKernel,
}

impl FixedFd3DKernel {
    /// Compile the sweep and its transpose for a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns the runtime compiler or module-load failure.
    pub fn new(_device: &CudaDevice) -> Result<Self> {
        let block = [
            u32::try_from(WORKGROUP[0]).expect("invariant: literal block dim fits u32"),
            u32::try_from(WORKGROUP[1]).expect("invariant: literal block dim fits u32"),
            u32::try_from(WORKGROUP[2]).expect("invariant: literal block dim fits u32"),
        ];
        Ok(Self {
            sweep: CudaMultiStorageKernel::new(
                "hephaestus-fixed-fd-3d-sweep",
                FIXED_FD_3D_KERNEL,
                "fixed_fd_sweep",
                &[0, 1],
                block,
                0,
            )?,
            adjoint: CudaMultiStorageKernel::new(
                "hephaestus-fixed-fd-3d-adjoint",
                FIXED_FD_3D_KERNEL,
                "fixed_fd_adjoint",
                &[0, 1],
                block,
                0,
            )?,
        })
    }

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the launch
    /// failure.
    pub fn sweep(
        &self,
        device: &CudaDevice,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
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
        MultiStorageKernel::<CudaDevice, FixedFd3DParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            &self.sweep,
            device,
            [
                CudaStorageBinding::new(0, input),
                CudaStorageBinding::new(1, output),
            ],
            params,
            grid,
        )
    }

    /// Sweep the transpose of the scheme in `params` along its axis,
    /// `upstream` into `grad`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the launch
    /// failure.
    pub fn adjoint(
        &self,
        device: &CudaDevice,
        upstream: &CudaBuffer<f32>,
        grad: &CudaBuffer<f32>,
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
        MultiStorageKernel::<CudaDevice, FixedFd3DParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            &self.adjoint,
            device,
            [
                CudaStorageBinding::new(0, upstream),
                CudaStorageBinding::new(1, grad),
            ],
            params,
            grid,
        )
    }
}
