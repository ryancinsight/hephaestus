//! Runtime-rank strided entry points.

use super::launch::{BinaryStridedLaunch, launch_binary_strided_expression, launch_unary_strided};
use super::types::StridedOperandDyn;
use super::validate::{broadcast_dyn_layout, validate_dyn_layout, validate_dyn_out};
use super::{StridedMeta, pad_shape_dyn, pad_usize_strides_dyn, to_u32};
use crate::CudaDevice;
use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, CudaC, DialectScalar, Result, TypedBinaryExpr, UnaryExpr,
};

fn binary_elementwise_strided_dyn_into_expression<T>(
    device: &CudaDevice,
    a: StridedOperandDyn<'_, T>,
    b: StridedOperandDyn<'_, T>,
    out: StridedOperandDyn<'_, T>,
    width: BlockWidth,
    operation: std::any::TypeId,
    expr: &'static str,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    validate_dyn_layout("binary left", a.buffer, a.layout)?;
    validate_dyn_layout("binary right", b.buffer, b.layout)?;
    let len = validate_dyn_out(out.buffer, out.layout)?;
    if len == 0 {
        return Ok(());
    }

    let a_layout = broadcast_dyn_layout(a.layout, out.layout.shape)?;
    let b_layout = broadcast_dyn_layout(b.layout, out.layout.shape)?;
    let meta = StridedMeta {
        shape: pad_shape_dyn(out.layout.shape)?,
        a_strides: a_layout.strides,
        b_strides: b_layout.strides,
        out_strides: pad_usize_strides_dyn(out.layout.strides)?,
        offsets: [
            a_layout.offset,
            b_layout.offset,
            to_u32(out.layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    launch_binary_strided_expression(
        device,
        BinaryStridedLaunch {
            a: a.buffer,
            b: b.buffer,
            out: out.buffer,
            meta,
            width,
            len,
            operation,
            expr,
        },
    )
}

/// Run `out[idx] = op(a[idx], b[idx])` over dynamic-rank logical output indices.
pub fn binary_elementwise_strided_dyn_into<Op, T>(
    device: &CudaDevice,
    a: StridedOperandDyn<'_, T>,
    b: StridedOperandDyn<'_, T>,
    out: StridedOperandDyn<'_, T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    binary_elementwise_strided_dyn_into_expression::<T>(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<CudaC>>::EXPR,
    )
}

/// Run a scalar-aware binary operation over dynamic-rank logical output
/// indices.
pub fn binary_elementwise_strided_dyn_typed_into<Op, T>(
    device: &CudaDevice,
    a: StridedOperandDyn<'_, T>,
    b: StridedOperandDyn<'_, T>,
    out: StridedOperandDyn<'_, T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: TypedBinaryExpr<CudaC, T>,
    T: DialectScalar<CudaC> + Pod,
{
    binary_elementwise_strided_dyn_into_expression::<T>(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<CudaC, T>>::EXPR,
    )
}

/// Run `out[idx] = op(a[idx])` over dynamic-rank logical output indices.
pub fn unary_elementwise_strided_dyn_into<Op, T>(
    device: &CudaDevice,
    a: StridedOperandDyn<'_, T>,
    out: StridedOperandDyn<'_, T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    validate_dyn_layout("unary input", a.buffer, a.layout)?;
    let len = validate_dyn_out(out.buffer, out.layout)?;
    if len == 0 {
        return Ok(());
    }

    let a_layout = broadcast_dyn_layout(a.layout, out.layout.shape)?;
    let meta = StridedMeta {
        shape: pad_shape_dyn(out.layout.shape)?,
        a_strides: a_layout.strides,
        b_strides: [0; 8],
        out_strides: pad_usize_strides_dyn(out.layout.strides)?,
        offsets: [
            a_layout.offset,
            0,
            to_u32(out.layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    launch_unary_strided::<Op, T>(device, a.buffer, out.buffer, meta, width, len)
}
