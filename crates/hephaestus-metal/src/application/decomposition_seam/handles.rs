use hephaestus_core::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, FullPivLuHandle,
    HessenbergHandle, LuHandle, QrHandle, Result, SchurHandle, SvdHandle, SymmetricEigenHandle,
    UduHandle,
};

use crate::application::decomposition::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition,
};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Column-pivoted QR result: the WGPU handle wrapped for this device.
pub struct MetalColPivQrDecomposition {
    pub(super) inner: GpuColPivQrDecomposition,
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
    pub(super) inner: GpuFullPivLuDecomposition,
    pub(super) factors: MetalBuffer<f32>,
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

/// Symmetric eigendecomposition result: the WGPU handle plus its factors
/// rewrapped for this device.
pub struct MetalSymmetricEigenDecomposition {
    pub(super) inner: GpuSymmetricEigenDecomposition,
    pub(super) eigenvalues: MetalBuffer<f32>,
    pub(super) eigenvectors: MetalBuffer<f32>,
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
    pub(super) inner: GpuSvdDecomposition,
    pub(super) u: MetalBuffer<f32>,
    pub(super) v: MetalBuffer<f32>,
    pub(super) singular_values: MetalBuffer<f32>,
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

/// Bunch–Kaufman result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalBunchKaufmanDecomposition {
    pub(super) inner: GpuBunchKaufmanDecomposition,
    pub(super) l: MetalBuffer<f32>,
    pub(super) d: MetalBuffer<f32>,
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
    pub(super) inner: GpuUduDecomposition,
    pub(super) u: MetalBuffer<f32>,
    pub(super) d: MetalBuffer<f32>,
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

/// Real Schur result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalRealSchur {
    pub(super) inner: GpuRealSchur,
    pub(super) q: MetalBuffer<f32>,
    pub(super) t: MetalBuffer<f32>,
}

impl SchurHandle<MetalDevice> for MetalRealSchur {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn q_buffer(&self) -> &MetalBuffer<f32> {
        &self.q
    }
    fn t_buffer(&self) -> &MetalBuffer<f32> {
        &self.t
    }
}

/// Hessenberg result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalHessenbergDecomposition {
    pub(super) inner: GpuHessenbergDecomposition,
    pub(super) q: MetalBuffer<f32>,
    pub(super) h: MetalBuffer<f32>,
}

impl HessenbergHandle<MetalDevice> for MetalHessenbergDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn q_buffer(&self) -> &MetalBuffer<f32> {
        &self.q
    }
    fn h_buffer(&self) -> &MetalBuffer<f32> {
        &self.h
    }
}

/// Bidiagonal result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalBidiagonalDecomposition {
    pub(super) inner: GpuBidiagonalDecomposition,
    pub(super) u: MetalBuffer<f32>,
    pub(super) b: MetalBuffer<f32>,
    pub(super) v: MetalBuffer<f32>,
}

impl BidiagonalHandle<MetalDevice> for MetalBidiagonalDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn u_buffer(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn b_buffer(&self) -> &MetalBuffer<f32> {
        &self.b
    }
    fn v_buffer(&self) -> &MetalBuffer<f32> {
        &self.v
    }
}

/// LU result: the WGPU handle plus its factors rewrapped for this device.
pub struct MetalLuDecomposition {
    pub(super) inner: GpuLuDecomposition,
    pub(super) factors: MetalBuffer<f32>,
}

/// QR result: the WGPU handle plus its `R` rewrapped for this device.
pub struct MetalQrDecomposition {
    pub(super) inner: GpuQrDecomposition,
    pub(super) r: MetalBuffer<f32>,
}

/// Cholesky result: the WGPU handle plus its `L` rewrapped for this device.
pub struct MetalCholeskyDecomposition {
    pub(super) inner: GpuCholesky,
    pub(super) lower: MetalBuffer<f32>,
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
