//! Elementwise compute dispatch.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{BinaryExpr, BlockWidth, DialectScalar, Result, UnaryExpr, Wgsl};
use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::{
    AbsOp, AddOp, CosOp, DivOp, ExpOp, IdentityOp, LnOp, MulOp, NegOp, PowOp, RecipOp, SinOp,
    SqrtOp, SubOp,
};

/// Run binary elementwise operation, allocating a new buffer.
#[inline]
pub fn binary_elementwise<Op, T>(
    device: &MetalDevice,
    lhs: &MetalBuffer<T>,
    rhs: &MetalBuffer<T>,
) -> Result<MetalBuffer<T>>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    let inner = wgpu_backend::binary_elementwise::<Op, T>(&device.inner, &lhs.inner, &rhs.inner)?;
    Ok(MetalBuffer { inner })
}

/// Run binary elementwise operation, writing into an existing buffer.
#[inline]
pub fn binary_elementwise_into<Op, T>(
    device: &MetalDevice,
    lhs: &MetalBuffer<T>,
    rhs: &MetalBuffer<T>,
    out: &MetalBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    wgpu_backend::binary_elementwise_into::<Op, T>(
        &device.inner,
        &lhs.inner,
        &rhs.inner,
        &out.inner,
        width,
    )
}

/// Run binary elementwise operation with a scalar, allocating a new buffer.
#[inline]
pub fn scalar_elementwise<Op, T>(
    device: &MetalDevice,
    lhs: &MetalBuffer<T>,
    scalar: T,
) -> Result<MetalBuffer<T>>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    let inner = wgpu_backend::scalar_elementwise::<Op, T>(&device.inner, &lhs.inner, scalar)?;
    Ok(MetalBuffer { inner })
}

/// Run binary elementwise operation with a scalar, writing into an existing buffer.
#[inline]
pub fn scalar_elementwise_into<Op, T>(
    device: &MetalDevice,
    lhs: &MetalBuffer<T>,
    scalar: T,
    out: &MetalBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    wgpu_backend::scalar_elementwise_into::<Op, T>(
        &device.inner,
        &lhs.inner,
        scalar,
        &out.inner,
        width,
    )
}

/// Run unary elementwise operation, allocating a new buffer.
#[inline]
pub fn unary_elementwise<Op, T>(
    device: &MetalDevice,
    lhs: &MetalBuffer<T>,
) -> Result<MetalBuffer<T>>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    let inner = wgpu_backend::unary_elementwise::<Op, T>(&device.inner, &lhs.inner)?;
    Ok(MetalBuffer { inner })
}

/// Run unary elementwise operation, writing into an existing buffer.
#[inline]
pub fn unary_elementwise_into<Op, T>(
    device: &MetalDevice,
    lhs: &MetalBuffer<T>,
    out: &MetalBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    wgpu_backend::unary_elementwise_into::<Op, T>(&device.inner, &lhs.inner, &out.inner, width)
}
