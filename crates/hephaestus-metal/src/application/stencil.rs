//! Metal 2D Laplacian stencil delegation.

use hephaestus_core::Result;
use hephaestus_wgpu as wgpu_backend;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

pub use hephaestus_core::{BoundaryCondition, Laplacian2DParams, LaplacianPolarity};

/// Compiled 2D Laplacian stencil using the native Metal-selected WGPU kernel.
#[derive(Debug)]
pub struct Laplacian2DKernel {
    inner: wgpu_backend::Laplacian2DKernel,
}

impl Laplacian2DKernel {
    /// Compile the stencil for a Metal device.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        Ok(Self {
            inner: wgpu_backend::Laplacian2DKernel::new(device.wgpu_device())?,
        })
    }

    /// Dispatch the stencil over Metal buffers.
    pub fn dispatch(
        &self,
        device: &MetalDevice,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()> {
        self.inner.dispatch(
            device.wgpu_device(),
            input.wgpu_buffer(),
            output.wgpu_buffer(),
            params,
        )
    }
}
