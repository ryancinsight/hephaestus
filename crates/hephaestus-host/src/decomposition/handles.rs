use hephaestus_core::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, FullPivLuHandle,
    HessenbergHandle, LuHandle, QrHandle, Result, SchurHandle, SvdHandle, SymmetricEigenHandle,
    UduHandle,
};

use super::solve_with;
use crate::{HostBuffer, HostDevice};

/// Partially pivoted LU on the host.
pub struct HostLu {
    pub(super) inner: leto_ops::LuDecomposition<f32>,
    pub(super) factors: HostBuffer<f32>,
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
    pub(super) inner: leto_ops::QrDecomposition<f32>,
    pub(super) r: HostBuffer<f32>,
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
    pub(super) inner: leto_ops::CholeskyDecomposition<f32>,
    pub(super) lower: HostBuffer<f32>,
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

/// Column-pivoted QR on the host.
pub struct HostColPivQr {
    pub(super) inner: leto_ops::ColPivQrDecomposition<f32>,
    pub(super) shape: (usize, usize),
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
    pub(super) inner: leto_ops::FullPivLuDecomposition<f32>,
    pub(super) order: usize,
    pub(super) factors: HostBuffer<f32>,
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

/// Symmetric Jacobi eigendecomposition on the host.
pub struct HostSymmetricEigen {
    pub(super) order: usize,
    pub(super) eigenvalues: HostBuffer<f32>,
    pub(super) eigenvectors: HostBuffer<f32>,
}

impl SymmetricEigenHandle<HostDevice> for HostSymmetricEigen {
    fn order(&self) -> usize {
        self.order
    }
    fn eigenvalues(&self) -> &HostBuffer<f32> {
        &self.eigenvalues
    }
    fn eigenvectors(&self) -> &HostBuffer<f32> {
        &self.eigenvectors
    }
}

/// SVD on the host.
pub struct HostSvd {
    pub(super) shape: (usize, usize),
    pub(super) u: HostBuffer<f32>,
    pub(super) v: HostBuffer<f32>,
    pub(super) singular_values: HostBuffer<f32>,
}

impl SvdHandle<HostDevice> for HostSvd {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn u(&self) -> &HostBuffer<f32> {
        &self.u
    }
    fn v(&self) -> &HostBuffer<f32> {
        &self.v
    }
    fn singular_values(&self) -> &HostBuffer<f32> {
        &self.singular_values
    }
}

/// Real Schur form on the host.
pub struct HostSchur {
    pub(super) order: usize,
    pub(super) q: HostBuffer<f32>,
    pub(super) t: HostBuffer<f32>,
}

impl SchurHandle<HostDevice> for HostSchur {
    fn order(&self) -> usize {
        self.order
    }
    fn q_buffer(&self) -> &HostBuffer<f32> {
        &self.q
    }
    fn t_buffer(&self) -> &HostBuffer<f32> {
        &self.t
    }
}

/// Hessenberg reduction on the host.
pub struct HostHessenberg {
    pub(super) order: usize,
    pub(super) q: HostBuffer<f32>,
    pub(super) h: HostBuffer<f32>,
}

impl HessenbergHandle<HostDevice> for HostHessenberg {
    fn order(&self) -> usize {
        self.order
    }
    fn q_buffer(&self) -> &HostBuffer<f32> {
        &self.q
    }
    fn h_buffer(&self) -> &HostBuffer<f32> {
        &self.h
    }
}

/// Bidiagonal reduction on the host.
pub struct HostBidiagonal {
    pub(super) shape: (usize, usize),
    pub(super) u: HostBuffer<f32>,
    pub(super) b: HostBuffer<f32>,
    pub(super) v: HostBuffer<f32>,
}

impl BidiagonalHandle<HostDevice> for HostBidiagonal {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn u_buffer(&self) -> &HostBuffer<f32> {
        &self.u
    }
    fn b_buffer(&self) -> &HostBuffer<f32> {
        &self.b
    }
    fn v_buffer(&self) -> &HostBuffer<f32> {
        &self.v
    }
}

/// Bunch–Kaufman factorization on the host.
pub struct HostBunchKaufman {
    pub(super) inner: leto_ops::BunchKaufmanDecomposition<f32>,
    pub(super) order: usize,
    pub(super) l: HostBuffer<f32>,
    pub(super) d: HostBuffer<f32>,
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
    pub(super) inner: leto_ops::UduDecomposition<f32>,
    pub(super) order: usize,
    pub(super) u: HostBuffer<f32>,
    pub(super) d: HostBuffer<f32>,
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
