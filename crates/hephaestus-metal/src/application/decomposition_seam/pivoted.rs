use hephaestus_core::{ColPivQrHandle, FullPivLuHandle, Result};

use crate::application::decomposition::{GpuColPivQrDecomposition, GpuFullPivLuDecomposition};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Column-pivoted QR result: the WGPU handle wrapped for this device.
pub struct MetalColPivQrDecomposition {
    pub(crate) inner: GpuColPivQrDecomposition,
}

impl ColPivQrHandle<MetalDevice> for MetalColPivQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn permutation(&self) -> &[usize] {
        self.inner.permutation()
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

/// Fully pivoted LU result: the WGPU handle plus its factors rewrapped
/// for this device.
pub struct MetalFullPivLuDecomposition {
    pub(crate) inner: GpuFullPivLuDecomposition,
    pub(crate) factors: MetalBuffer<f32>,
}

impl FullPivLuHandle<MetalDevice> for MetalFullPivLuDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn factors(&self) -> &MetalBuffer<f32> {
        &self.factors
    }
    fn row_permutation(&self) -> &[usize] {
        self.inner.row_permutation()
    }
    fn col_permutation(&self) -> &[usize] {
        self.inner.col_permutation()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}
