use hephaestus_core::{BunchKaufmanHandle, Result, UduHandle};

use crate::application::decomposition::{GpuBunchKaufmanDecomposition, GpuUduDecomposition};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Bunch–Kaufman result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalBunchKaufmanDecomposition {
    pub(crate) inner: GpuBunchKaufmanDecomposition,
    pub(crate) l: MetalBuffer<f32>,
    pub(crate) d: MetalBuffer<f32>,
}

impl BunchKaufmanHandle<MetalDevice> for MetalBunchKaufmanDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn l_buffer(&self) -> &MetalBuffer<f32> {
        &self.l
    }
    fn d_buffer(&self) -> &MetalBuffer<f32> {
        &self.d
    }
    fn permutation(&self) -> &[usize] {
        self.inner.permutation()
    }
}

/// `U·D·Uᵀ` result: the WGPU handle plus its factors rewrapped for this
/// device.
pub struct MetalUduDecomposition {
    pub(crate) inner: GpuUduDecomposition,
    pub(crate) u: MetalBuffer<f32>,
    pub(crate) d: MetalBuffer<f32>,
}

impl UduHandle<MetalDevice> for MetalUduDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn u_buffer(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn d_buffer(&self) -> &MetalBuffer<f32> {
        &self.d
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
