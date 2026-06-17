//! Linear algebra operations.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::Result;
use hephaestus_wgpu as wgpu_backend;

/// Matrix multiplication of two 2D matrices, allocating a new buffer.
#[inline]
pub fn matmul(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 2>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::matmul(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
    )?;
    Ok(MetalBuffer { inner })
}

/// Matrix multiplication of two 2D matrices, writing into an existing buffer.
#[inline]
pub fn matmul_into(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 2>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 2>,
    out: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<()> {
    wgpu_backend::matmul_into(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
        crate::application::strided::to_wgpu_strided(out),
    )
}

/// Batched matrix multiplication of two 3D tensors, allocating a new buffer.
#[inline]
pub fn batched_matmul(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 3>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 3>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::batched_matmul(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
    )?;
    Ok(MetalBuffer { inner })
}

/// Batched matrix multiplication of two 3D tensors, writing into an existing buffer.
#[inline]
pub fn batched_matmul_into(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 3>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 3>,
    out: crate::application::strided::StridedOperand<'_, f32, 3>,
) -> Result<()> {
    wgpu_backend::batched_matmul_into(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
        crate::application::strided::to_wgpu_strided(out),
    )
}

/// Kronecker product of two 2D matrices, allocating a new buffer.
#[inline]
pub fn kron(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 2>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::kron(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
    )?;
    Ok(MetalBuffer { inner })
}

/// Kronecker product of two 2D matrices, writing into an existing buffer.
#[inline]
pub fn kron_into(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 2>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 2>,
    out: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<()> {
    wgpu_backend::kron_into(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
        crate::application::strided::to_wgpu_strided(out),
    )
}

/// Dot product of two 1D vectors, allocating a new buffer.
#[inline]
pub fn dot(
    device: &MetalDevice,
    lhs: crate::application::strided::StridedOperand<'_, f32, 1>,
    rhs: crate::application::strided::StridedOperand<'_, f32, 1>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::dot(
        &device.inner,
        crate::application::strided::to_wgpu_strided(lhs),
        crate::application::strided::to_wgpu_strided(rhs),
    )?;
    Ok(MetalBuffer { inner })
}

/// Matrix trace, allocating a new buffer.
#[inline]
pub fn trace(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::trace(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// L1 norm of a 2D matrix, allocating a new buffer.
#[inline]
pub fn norm_l1(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::norm_l1(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// L2 norm of a 2D matrix, allocating a new buffer.
#[inline]
pub fn norm_l2(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::norm_l2(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Max norm of a 2D matrix, allocating a new buffer.
#[inline]
pub fn norm_max(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::norm_max(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Integer matrix power, allocating a new buffer.
#[inline]
pub fn matpow(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
    exponent: u32,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::matpow(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
        exponent,
    )?;
    Ok(MetalBuffer { inner })
}

/// Matrix exponential, allocating a new buffer.
#[inline]
pub fn matexp(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::matexp(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Moore-Penrose pseudoinverse, allocating a new buffer.
#[inline]
pub fn pinv(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::pinv(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Matrix determinant, allocating a new buffer.
#[inline]
pub fn det(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<MetalBuffer<f32>> {
    let inner = wgpu_backend::det(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )?;
    Ok(MetalBuffer { inner })
}

/// Numerical rank of a 2D matrix.
#[inline]
pub fn matrix_rank(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
) -> Result<usize> {
    wgpu_backend::matrix_rank(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
    )
}

/// Numerical rank of a 2D matrix with a specific relative tolerance.
#[inline]
pub fn matrix_rank_with_tolerance(
    device: &MetalDevice,
    matrix: crate::application::strided::StridedOperand<'_, f32, 2>,
    tolerance: f32,
) -> Result<usize> {
    wgpu_backend::matrix_rank_with_tolerance(
        &device.inner,
        crate::application::strided::to_wgpu_strided(matrix),
        tolerance,
    )
}
