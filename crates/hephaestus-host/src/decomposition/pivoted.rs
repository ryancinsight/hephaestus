use hephaestus_core::{ColPivQrHandle, FullPivLuHandle, Result};

use crate::{HostBuffer, HostDevice};

use super::helpers::solve_with;

/// Column-pivoted QR on the host.
pub struct HostColPivQr {
    pub(crate) inner: leto_ops::ColPivQrDecomposition<f32>,
    pub(crate) shape: (usize, usize),
}

impl ColPivQrHandle<HostDevice> for HostColPivQr {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn permutation(&self) -> &[usize] {
        self.inner.permutation()
    }
    fn solve_least_squares(
        &self,
        device: &HostDevice,
        rhs: &HostBuffer<f32>,
    ) -> Result<HostBuffer<f32>> {
        solve_with(device, rhs, |view| self.inner.solve_least_squares(&view))
    }
}

/// Fully pivoted LU on the host.
pub struct HostFullPivLu {
    pub(crate) inner: leto_ops::FullPivLuDecomposition<f32>,
    pub(crate) order: usize,
    pub(crate) factors: HostBuffer<f32>,
}

impl FullPivLuHandle<HostDevice> for HostFullPivLu {
    fn order(&self) -> usize {
        self.order
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn factors(&self) -> &HostBuffer<f32> {
        &self.factors
    }
    fn row_permutation(&self) -> &[usize] {
        self.inner.row_permutation()
    }
    fn col_permutation(&self) -> &[usize] {
        self.inner.col_permutation()
    }
    fn solve(&self, device: &HostDevice, rhs: &HostBuffer<f32>) -> Result<HostBuffer<f32>> {
        solve_with(device, rhs, |view| self.inner.solve(&view))
    }
}
