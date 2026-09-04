//! Leto as a decomposition-seam implementor (ADR 0046 / ADR 0039 §3).
//!
//! [`HostDecompositionOps`] adapts leto-ops' fifteen decomposition entry
//! points onto [`DecompositionOps<HostDevice>`](hephaestus_core::DecompositionOps), so the CPU substrate joins
//! the same role trait the GPU backends implement and the conformance
//! suite's decomposition clauses run on the host pair. Factor storage
//! conventions follow the seam contracts pinned by the clause suite
//! (packed LU factors, m×n row-major `R`, column-convention singular and
//! eigen vectors, dense block-diagonal Bunch–Kaufman `D`, length-`n` UDU
//! diagonal).

use eunomia::Pod;
use hephaestus_core::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, ComputeDevice,
    DecompositionOps, FullPivLuHandle, HephaestusError, HessenbergHandle, LuHandle, QrHandle,
    Result, SchurHandle, StridedView, SvdHandle, SymmetricEigenHandle, UduHandle,
};
use leto::Layout;

use crate::{HostBuffer, HostDevice};

/// Dense decompositions for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDecompositionOps;

fn map_leto_err<E: core::fmt::Display>(error: E) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("host decomposition failed: {error}"),
    }
}

fn buffer_of<T: Pod>(device: &HostDevice, values: &[T]) -> Result<HostBuffer<T>> {
    device.upload(values)
}

fn array_buffer(device: &HostDevice, array: &leto::Array2<f32>) -> Result<HostBuffer<f32>> {
    buffer_of(device, leto::Storage::as_slice(array.storage()))
}

/// Run `operation` over the input view's host memory as a leto view.
fn with_view<R>(
    input: &StridedView<'_, HostBuffer<f32>, 2>,
    operation: impl FnOnce(leto::ArrayView<'_, f32, 2>) -> R,
) -> R {
    let cells = input.buffer.read();
    operation(leto::ArrayView::<f32, 2>::new(*input.layout, &cells))
}

/// Solve through a leto handle: download the rhs, apply, re-upload.
fn solve_with<E: core::fmt::Display>(
    device: &HostDevice,
    rhs: &HostBuffer<f32>,
    solve: impl FnOnce(leto::ArrayView<'_, f32, 1>) -> core::result::Result<leto::Array1<f32>, E>,
) -> Result<HostBuffer<f32>> {
    let cells = rhs.read();
    let layout =
        Layout::c_contiguous([cells.len()]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("host rhs layout rejected: {error}"),
        })?;
    let solution = solve(leto::ArrayView::<f32, 1>::new(layout, &cells)).map_err(map_leto_err)?;
    buffer_of(device, leto::Storage::as_slice(solution.storage()))
}

/// Partially pivoted LU on the host.
pub struct HostLu {
    inner: leto_ops::LuDecomposition<f32>,
    factors: HostBuffer<f32>,
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
    inner: leto_ops::QrDecomposition<f32>,
    r: HostBuffer<f32>,
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
    inner: leto_ops::CholeskyDecomposition<f32>,
    lower: HostBuffer<f32>,
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
    inner: leto_ops::ColPivQrDecomposition<f32>,
    shape: (usize, usize),
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
    inner: leto_ops::FullPivLuDecomposition<f32>,
    order: usize,
    factors: HostBuffer<f32>,
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
    order: usize,
    eigenvalues: HostBuffer<f32>,
    eigenvectors: HostBuffer<f32>,
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
    shape: (usize, usize),
    u: HostBuffer<f32>,
    v: HostBuffer<f32>,
    singular_values: HostBuffer<f32>,
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
    order: usize,
    q: HostBuffer<f32>,
    t: HostBuffer<f32>,
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
    order: usize,
    q: HostBuffer<f32>,
    h: HostBuffer<f32>,
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
    shape: (usize, usize),
    u: HostBuffer<f32>,
    b: HostBuffer<f32>,
    v: HostBuffer<f32>,
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
    inner: leto_ops::BunchKaufmanDecomposition<f32>,
    order: usize,
    l: HostBuffer<f32>,
    d: HostBuffer<f32>,
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
    inner: leto_ops::UduDecomposition<f32>,
    order: usize,
    u: HostBuffer<f32>,
    d: HostBuffer<f32>,
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

impl DecompositionOps<HostDevice> for HostDecompositionOps {
    type Lu<'op> = HostLu;
    type Qr<'op> = HostQr;
    type Cholesky<'op> = HostCholesky;
    type ColPivQr<'op> = HostColPivQr;
    type FullPivLu<'op> = HostFullPivLu;
    type SymmetricEigen<'op> = HostSymmetricEigen;
    type Svd<'op> = HostSvd;
    type Schur<'op> = HostSchur;
    type Hessenberg<'op> = HostHessenberg;
    type Bidiagonal<'op> = HostBidiagonal;
    type BunchKaufman<'op> = HostBunchKaufman;
    type Udu<'op> = HostUdu;

    fn lu<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>> {
        let inner =
            with_view(&input, |view| leto_ops::lu_decompose(&view)).map_err(map_leto_err)?;
        let factors = array_buffer(device, inner.factors())?;
        Ok(HostLu { inner, factors })
    }

    fn qr<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>> {
        let inner =
            with_view(&input, |view| leto_ops::qr_decompose(&view)).map_err(map_leto_err)?;
        let r = array_buffer(device, &inner.r())?;
        Ok(HostQr { inner, r })
    }

    fn cholesky<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        let inner =
            with_view(&input, |view| leto_ops::cholesky_decompose(&view)).map_err(map_leto_err)?;
        let lower = array_buffer(device, inner.lower())?;
        Ok(HostCholesky { inner, lower })
    }

    fn col_piv_qr<'op>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>> {
        let shape = (input.layout.shape()[0], input.layout.shape()[1]);
        let inner = with_view(&input, |view| leto_ops::col_piv_qr(&view)).map_err(map_leto_err)?;
        Ok(HostColPivQr { inner, shape })
    }

    fn full_piv_lu<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_view(&input, |view| leto_ops::full_piv_lu(&view)).map_err(map_leto_err)?;
        let factors = buffer_of(device, inner.lu_factors())?;
        Ok(HostFullPivLu {
            inner,
            order,
            factors,
        })
    }

    fn symmetric_eigen<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::SymmetricEigen<'op>> {
        let inner = with_view(&input, |view| leto_ops::symmetric_eigen_jacobi(&view))
            .map_err(map_leto_err)?;
        Ok(HostSymmetricEigen {
            order: inner.eigenvalues.len(),
            eigenvalues: buffer_of(device, &inner.eigenvalues)?,
            eigenvectors: array_buffer(device, &inner.eigenvectors)?,
        })
    }

    fn symmetric_eigenvalues(
        &self,
        device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let values = with_view(&input, |view| leto_ops::symmetric_eigenvalues_jacobi(&view))
            .map_err(map_leto_err)?;
        buffer_of(device, &values)
    }

    fn svd<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Svd<'op>> {
        let shape = (input.layout.shape()[0], input.layout.shape()[1]);
        let inner =
            with_view(&input, |view| leto_ops::svd_decompose(&view)).map_err(map_leto_err)?;
        Ok(HostSvd {
            shape,
            u: array_buffer(device, &inner.left_singular_vectors)?,
            v: array_buffer(device, &inner.right_singular_vectors)?,
            singular_values: buffer_of(device, &inner.singular_values)?,
        })
    }

    fn singular_values(
        &self,
        device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let values =
            with_view(&input, |view| leto_ops::singular_values(&view)).map_err(map_leto_err)?;
        buffer_of(device, &values)
    }

    fn eigenvalues(
        &self,
        device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<eunomia::Complex<f32>>> {
        let values =
            with_view(&input, |view| leto_ops::eigenvalues(&view)).map_err(map_leto_err)?;
        buffer_of(device, &values)
    }

    fn schur<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Schur<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_view(&input, |view| leto_ops::schur(&view)).map_err(map_leto_err)?;
        Ok(HostSchur {
            order,
            q: array_buffer(device, &inner.q())?,
            t: array_buffer(device, &inner.t())?,
        })
    }

    fn hessenberg<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Hessenberg<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_view(&input, |view| leto_ops::hessenberg(&view)).map_err(map_leto_err)?;
        Ok(HostHessenberg {
            order,
            q: array_buffer(device, inner.q())?,
            h: array_buffer(device, inner.h())?,
        })
    }

    fn bidiagonalize<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Bidiagonal<'op>> {
        let shape = (input.layout.shape()[0], input.layout.shape()[1]);
        let inner =
            with_view(&input, |view| leto_ops::bidiagonalize(&view)).map_err(map_leto_err)?;
        Ok(HostBidiagonal {
            shape,
            u: array_buffer(device, inner.u())?,
            b: array_buffer(device, inner.b())?,
            v: array_buffer(device, inner.v())?,
        })
    }

    fn bunch_kaufman<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::BunchKaufman<'op>> {
        let order = input.layout.shape()[0];
        let inner =
            with_view(&input, |view| leto_ops::bunch_kaufman(&view)).map_err(map_leto_err)?;
        Ok(HostBunchKaufman {
            order,
            l: array_buffer(device, &inner.l())?,
            d: array_buffer(device, &inner.d())?,
            inner,
        })
    }

    fn udu<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Udu<'op>> {
        let order = input.layout.shape()[0];
        let inner =
            with_view(&input, |view| leto_ops::udu_decompose(&view)).map_err(map_leto_err)?;
        Ok(HostUdu {
            order,
            u: array_buffer(device, &inner.u())?,
            d: buffer_of(device, inner.diagonal())?,
            inner,
        })
    }
}
