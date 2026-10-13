use hephaestus_core::{CholeskyHandle, LuHandle, QrHandle, Result};

use crate::application::decomposition::{GpuCholesky, GpuLuDecomposition, GpuQrDecomposition};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// LU result: the WGPU handle plus its factors rewrapped for this device.
pub struct MetalLuDecomposition {
    pub(crate) inner: GpuLuDecomposition,
    pub(crate) factors: MetalBuffer<f32>,
}

/// QR result: the WGPU handle plus its `R` rewrapped for this device.
pub struct MetalQrDecomposition {
    pub(crate) inner: GpuQrDecomposition,
    pub(crate) r: MetalBuffer<f32>,
}

/// Cholesky result: the WGPU handle plus its `L` rewrapped for this device.
pub struct MetalCholeskyDecomposition {
    pub(crate) inner: GpuCholesky,
    pub(crate) lower: MetalBuffer<f32>,
}

impl LuHandle<MetalDevice> for MetalLuDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn factors(&self) -> &MetalBuffer<f32> {
        &self.factors
    }
    fn pivots(&self) -> &[usize] {
        self.inner.pivots()
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

impl QrHandle<MetalDevice> for MetalQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn r_buffer(&self) -> &MetalBuffer<f32> {
        &self.r
    }
    fn solve_least_squares(
        &self,
        device: &MetalDevice,
        rhs: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve_least_squares(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

impl CholeskyHandle<MetalDevice> for MetalCholeskyDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn lower(&self) -> &MetalBuffer<f32> {
        &self.lower
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}
