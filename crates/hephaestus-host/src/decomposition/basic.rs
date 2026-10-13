use hephaestus_core::{CholeskyHandle, LuHandle, QrHandle, Result};

use crate::{HostBuffer, HostDevice};

use super::solve::solve_with;

/// Partially pivoted LU on the host.
pub struct HostLu {
    pub(crate) inner: leto_ops::LuDecomposition<f32>,
    pub(crate) factors: HostBuffer<f32>,
}

impl LuHandle<HostDevice> for HostLu {
    fn order(&self) -> usize {
        self.inner.dim()
    }
    fn factors(&self) -> &HostBuffer<f32> {
        &self.factors
    }
    fn pivots(&self) -> &[usize] {
        self.inner.pivots()
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &HostDevice, rhs: &HostBuffer<f32>) -> Result<HostBuffer<f32>> {
        solve_with(device, rhs, |view| self.inner.solve(&view))
    }
}

/// Householder QR on the host.
pub struct HostQr {
    pub(crate) inner: leto_ops::QrDecomposition<f32>,
    pub(crate) r: HostBuffer<f32>,
}

impl QrHandle<HostDevice> for HostQr {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn r_buffer(&self) -> &HostBuffer<f32> {
        &self.r
    }
    fn solve_least_squares(
        &self,
        device: &HostDevice,
        rhs: &HostBuffer<f32>,
    ) -> Result<HostBuffer<f32>> {
        solve_with(device, rhs, |view| self.inner.solve_least_squares(&view))
    }
}

/// Cholesky on the host.
pub struct HostCholesky {
    pub(crate) inner: leto_ops::CholeskyDecomposition<f32>,
    pub(crate) lower: HostBuffer<f32>,
}

impl CholeskyHandle<HostDevice> for HostCholesky {
    fn order(&self) -> usize {
        self.inner.dim()
    }
    fn lower(&self) -> &HostBuffer<f32> {
        &self.lower
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &HostDevice, rhs: &HostBuffer<f32>) -> Result<HostBuffer<f32>> {
        solve_with(device, rhs, |view| self.inner.solve(&view))
    }
}
