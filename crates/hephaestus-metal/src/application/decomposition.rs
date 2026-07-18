//! Matrix decompositions.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::Result;
use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition,
};

/// LU decomposition with partial pivoting.
#[inline]
pub fn lu_decompose(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    wgpu_backend::lu_decompose(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Blocked LU decomposition.
#[inline]
pub fn lu_decompose_blocked(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    wgpu_backend::lu_decompose_blocked(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// QR decomposition.
#[inline]
pub fn qr_decompose(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    wgpu_backend::qr_decompose(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Blocked QR decomposition.
#[inline]
pub fn qr_decompose_blocked(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    wgpu_backend::qr_decompose_blocked(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Cholesky decomposition.
#[inline]
pub fn cholesky_decompose(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    wgpu_backend::cholesky_decompose(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Blocked Cholesky decomposition.
#[inline]
pub fn cholesky_decompose_blocked(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    wgpu_backend::cholesky_decompose_blocked(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Bunch-Kaufman decomposition.
#[inline]
pub fn bunch_kaufman(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuBunchKaufmanDecomposition> {
    wgpu_backend::bunch_kaufman(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// UDU decomposition.
#[inline]
pub fn udu_decompose(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuUduDecomposition> {
    wgpu_backend::udu_decompose(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Column-pivoted QR decomposition.
#[inline]
pub fn col_piv_qr(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuColPivQrDecomposition> {
    wgpu_backend::col_piv_qr(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Full-pivoted LU decomposition.
#[inline]
pub fn full_piv_lu(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuFullPivLuDecomposition> {
    wgpu_backend::full_piv_lu(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Upper Hessenberg reduction.
#[inline]
pub fn hessenberg(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuHessenbergDecomposition> {
    wgpu_backend::hessenberg(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Golub-Kahan bidiagonalization.
#[inline]
pub fn bidiagonalize(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuBidiagonalDecomposition> {
    wgpu_backend::bidiagonalize(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Thin SVD decomposition.
#[inline]
pub fn svd_decompose(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuSvdDecomposition> {
    wgpu_backend::svd_decompose(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Rank-revealing SVD decomposition.
#[inline]
pub fn svd_rank_revealing(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuSvdDecomposition> {
    wgpu_backend::svd_rank_revealing(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Singular values of a 2D matrix.
#[inline]
pub fn singular_values(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::singular_values(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Symmetric eigendecomposition.
#[inline]
pub fn symmetric_eigen_jacobi(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuSymmetricEigenDecomposition> {
    wgpu_backend::symmetric_eigen_jacobi(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Symmetric eigenvalues only.
#[inline]
pub fn symmetric_eigenvalues_jacobi(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::symmetric_eigenvalues_jacobi(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Eigenvalues of a general matrix.
#[inline]
pub fn eigenvalues(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<eunomia::Complex<f32>>> {
    let inner = wgpu_backend::eigenvalues(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Real Schur decomposition.
#[inline]
pub fn schur(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<GpuRealSchur> {
    wgpu_backend::schur(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}
