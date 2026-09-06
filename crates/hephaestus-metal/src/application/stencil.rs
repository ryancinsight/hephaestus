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

/// Provider-owned implementation of [`hephaestus_core::StencilOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalStencilOps;

impl hephaestus_core::StencilOps<MetalDevice> for MetalStencilOps {
    type Laplacian2D = Laplacian2DKernel;

    fn prepare_laplacian_2d(&self, device: &MetalDevice) -> Result<Self::Laplacian2D> {
        Laplacian2DKernel::new(device)
    }

    fn laplacian_2d_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::Laplacian2D,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()> {
        kernel.dispatch(device, input, output, params)
    }
}

pub use hephaestus_core::{Staggered3DParams, StaggeredAxis};

/// Compiled 3-D staggered pair using the native Metal-selected WGPU kernels.
#[derive(Debug)]
pub struct Staggered3DKernel {
    inner: wgpu_backend::Staggered3DKernel,
}

impl Staggered3DKernel {
    /// Compile the staggered pair for a Metal device.
    ///
    /// # Errors
    ///
    /// Returns the WGPU backend's kernel compilation failure.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        Ok(Self {
            inner: wgpu_backend::Staggered3DKernel::new(device.wgpu_device())?,
        })
    }

    /// Dispatch the gradient over Metal buffers.
    ///
    /// # Errors
    ///
    /// Returns the WGPU backend's dispatch failure.
    pub fn gradient(
        &self,
        device: &MetalDevice,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        self.inner.gradient(
            device.wgpu_device(),
            input.wgpu_buffer(),
            output.wgpu_buffer(),
            params,
        )
    }

    /// Dispatch the divergence over Metal buffers.
    ///
    /// # Errors
    ///
    /// Returns the WGPU backend's dispatch failure.
    pub fn divergence(
        &self,
        device: &MetalDevice,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        self.inner.divergence(
            device.wgpu_device(),
            input.wgpu_buffer(),
            output.wgpu_buffer(),
            params,
        )
    }
}

/// Provider-owned implementation of [`hephaestus_core::Staggered3DOps`] for
/// Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalStaggered3DOps;

impl hephaestus_core::Staggered3DOps<MetalDevice> for MetalStaggered3DOps {
    type Staggered3D = Staggered3DKernel;

    fn prepare_staggered_3d(&self, device: &MetalDevice) -> Result<Self::Staggered3D> {
        Staggered3DKernel::new(device)
    }

    fn staggered_gradient_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::Staggered3D,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        kernel.gradient(device, input, output, params)
    }

    fn staggered_divergence_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::Staggered3D,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        kernel.divergence(device, input, output, params)
    }
}
