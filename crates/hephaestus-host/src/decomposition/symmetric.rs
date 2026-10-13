use hephaestus_core::{BunchKaufmanHandle, Result, UduHandle};

use crate::{HostBuffer, HostDevice};

use super::helpers::solve_with;

/// Bunch–Kaufman factorization on the host.
pub struct HostBunchKaufman {
    pub(crate) inner: leto_ops::BunchKaufmanDecomposition<f32>,
    pub(crate) order: usize,
    pub(crate) l: HostBuffer<f32>,
    pub(crate) d: HostBuffer<f32>,
}

impl BunchKaufmanHandle<HostDevice> for HostBunchKaufman {
    fn order(&self) -> usize {
        self.order
    }
    fn l_buffer(&self) -> &HostBuffer<f32> {
        &self.l
    }
    fn d_buffer(&self) -> &HostBuffer<f32> {
        &self.d
    }
    fn permutation(&self) -> &[usize] {
        self.inner.permutation()
    }
}

/// `U·D·Uᵀ` factorization on the host.
pub struct HostUdu {
    pub(crate) inner: leto_ops::UduDecomposition<f32>,
    pub(crate) order: usize,
    pub(crate) u: HostBuffer<f32>,
    pub(crate) d: HostBuffer<f32>,
}

impl UduHandle<HostDevice> for HostUdu {
    fn order(&self) -> usize {
        self.order
    }
    fn u_buffer(&self) -> &HostBuffer<f32> {
        &self.u
    }
    fn d_buffer(&self) -> &HostBuffer<f32> {
        &self.d
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &HostDevice, rhs: &HostBuffer<f32>) -> Result<HostBuffer<f32>> {
        solve_with(device, rhs, |view| self.inner.solve(&view))
    }
}
