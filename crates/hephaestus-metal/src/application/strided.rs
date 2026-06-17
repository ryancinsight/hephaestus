//! Strided layout wrappers.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{BlockWidth, Result};
use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::MAX_STRIDED_RANK;

/// A device buffer paired with the leto layout describing its logical view.
#[derive(Clone, Copy)]
pub struct StridedOperand<'a, T, const N: usize> {
    /// The device buffer.
    pub buffer: &'a MetalBuffer<T>,
    /// The logical layout over that buffer.
    pub layout: &'a leto::Layout<N>,
}

#[inline]
pub(crate) fn to_wgpu_strided<'a, T, const N: usize>(
    op: StridedOperand<'a, T, N>,
) -> wgpu_backend::StridedOperand<'a, T, N> {
    wgpu_backend::StridedOperand {
        buffer: &op.buffer.inner,
        layout: op.layout,
    }
}

/// Run binary elementwise operation on strided operands, allocating a new buffer.
#[inline]
pub fn binary_elementwise_strided<'a, Op, T, const N: usize>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, N>,
    rhs: StridedOperand<'a, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::BinaryWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod,
{
    let inner = wgpu_backend::binary_elementwise_strided::<Op, T, N>(
        &device.inner,
        to_wgpu_strided(lhs),
        to_wgpu_strided(rhs),
        output_shape,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Run binary elementwise operation on strided operands, writing into an existing buffer.
#[inline]
pub fn binary_elementwise_strided_into<'a, Op, T, const N: usize>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, N>,
    rhs: StridedOperand<'a, T, N>,
    out: StridedOperand<'a, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: wgpu_backend::BinaryWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod,
{
    wgpu_backend::binary_elementwise_strided_into::<Op, T, N>(
        &device.inner,
        to_wgpu_strided(lhs),
        to_wgpu_strided(rhs),
        to_wgpu_strided(out),
        width,
    )
}

/// Run binary elementwise operation with a scalar on strided operands, allocating a new buffer.
#[inline]
pub fn scalar_elementwise_strided<'a, Op, T, const N: usize>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, N>,
    scalar: T,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::BinaryWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod,
{
    let inner = wgpu_backend::scalar_elementwise_strided::<Op, T, N>(
        &device.inner,
        to_wgpu_strided(lhs),
        scalar,
        output_shape,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Run binary elementwise operation with a scalar on strided operands, writing into an existing buffer.
#[inline]
pub fn scalar_elementwise_strided_into<'a, Op, T, const N: usize>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, N>,
    scalar: T,
    out: StridedOperand<'a, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: wgpu_backend::BinaryWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod,
{
    wgpu_backend::scalar_elementwise_strided_into::<Op, T, N>(
        &device.inner,
        to_wgpu_strided(lhs),
        scalar,
        to_wgpu_strided(out),
        width,
    )
}

/// Run unary elementwise operation on a strided operand, allocating a new buffer.
#[inline]
pub fn unary_elementwise_strided<'a, Op, T, const N: usize>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::UnaryWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod,
{
    let inner = wgpu_backend::unary_elementwise_strided::<Op, T, N>(
        &device.inner,
        to_wgpu_strided(lhs),
        output_shape,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Run unary elementwise operation on a strided operand, writing into an existing buffer.
#[inline]
pub fn unary_elementwise_strided_into<'a, Op, T, const N: usize>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, N>,
    out: StridedOperand<'a, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: wgpu_backend::UnaryWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod,
{
    wgpu_backend::unary_elementwise_strided_into::<Op, T, N>(
        &device.inner,
        to_wgpu_strided(lhs),
        to_wgpu_strided(out),
        width,
    )
}
