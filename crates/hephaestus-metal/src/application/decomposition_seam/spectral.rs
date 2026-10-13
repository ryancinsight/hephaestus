use hephaestus_core::{SvdHandle, SymmetricEigenHandle};

use crate::application::decomposition::{GpuSvdDecomposition, GpuSymmetricEigenDecomposition};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Symmetric eigendecomposition result: the WGPU handle plus its factors
/// rewrapped for this device.
pub struct MetalSymmetricEigenDecomposition {
    pub(crate) inner: GpuSymmetricEigenDecomposition,
    pub(crate) eigenvalues: MetalBuffer<f32>,
    pub(crate) eigenvectors: MetalBuffer<f32>,
}

impl SymmetricEigenHandle<MetalDevice> for MetalSymmetricEigenDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn eigenvalues(&self) -> &MetalBuffer<f32> {
        &self.eigenvalues
    }
    fn eigenvectors(&self) -> &MetalBuffer<f32> {
        &self.eigenvectors
    }
}

/// SVD result: the WGPU handle plus its factors rewrapped for this
/// device.
pub struct MetalSvdDecomposition {
    pub(crate) inner: GpuSvdDecomposition,
    pub(crate) u: MetalBuffer<f32>,
    pub(crate) v: MetalBuffer<f32>,
    pub(crate) singular_values: MetalBuffer<f32>,
}

impl SvdHandle<MetalDevice> for MetalSvdDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn u(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn v(&self) -> &MetalBuffer<f32> {
        &self.v
    }
    fn singular_values(&self) -> &MetalBuffer<f32> {
        &self.singular_values
    }
}
