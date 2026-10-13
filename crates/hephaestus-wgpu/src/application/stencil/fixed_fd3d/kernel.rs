use hephaestus_core::{DispatchGrid, FixedFd3DParams, HephaestusError, MultiStorageKernel, Result};

use crate::application::storage_kernel::{
    WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::source::{FIXED_FD_3D_SHADER, WORKGROUP};

/// Compiled fixed-scheme sweep kernel.
///
/// Monomorphized at construction and reused across dispatches on the same
/// device; the scheme rides in each dispatch's parameter block.
#[derive(Debug)]
pub struct FixedFd3DKernel {
    sweep: WgslMultiStorageKernel,
    adjoint: WgslMultiStorageKernel,
}

impl FixedFd3DKernel {
    /// Compile the sweep and its transpose for a device.
    ///
    /// # Errors
    ///
    /// Returns `HephaestusError::DispatchFailed` when the WGSL source or
    /// binding layout is rejected by the device.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        let bindings = [
            WgslStorageBindingLayout::read_only(0),
            WgslStorageBindingLayout::read_write(2),
        ];
        Ok(Self {
            sweep: WgslMultiStorageKernel::new(
                device,
                "hephaestus-fixed-fd-3d-sweep",
                FIXED_FD_3D_SHADER,
                "fixed_fd_sweep",
                &bindings,
                1,
            )?,
            adjoint: WgslMultiStorageKernel::new(
                device,
                "hephaestus-fixed-fd-3d-adjoint",
                FIXED_FD_3D_SHADER,
                "fixed_fd_adjoint",
                &bindings,
                1,
            )?,
        })
    }

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the backend
    /// dispatch failure.
    pub fn sweep(
        &self,
        device: &WgpuDevice,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_storage(input.len, output.len)?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.output_dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        self.sweep.dispatch(
            device,
            [
                WgslStorageBinding::new(0, input),
                WgslStorageBinding::new(2, output),
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
    /// Returns a storage-length mismatch against either grid, or the backend
    /// dispatch failure.
    pub fn adjoint(
        &self,
        device: &WgpuDevice,
        upstream: &WgpuBuffer<f32>,
        grad: &WgpuBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_adjoint_storage(upstream.len, grad.len)?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        self.adjoint.dispatch(
            device,
            [
                WgslStorageBinding::new(0, upstream),
                WgslStorageBinding::new(2, grad),
            ],
            params,
            grid,
        )
    }
}
