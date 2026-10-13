//! Const-rank strided entry points.

use super::launch::{BinaryStridedLaunch, launch_binary_strided_expression, launch_unary_strided};
use super::meta::{binary_strided_meta, unary_strided_meta};
use super::types::StridedOperand;
use super::{MAX_STRIDED_RANK, map_layout_err};
use crate::CudaDevice;
use crate::infrastructure::buffer::CudaBuffer;
use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, CudaC, DialectScalar, Result, TypedBinaryExpr, UnaryExpr,
};
use leto::Layout;

fn binary_elementwise_strided_into_expression<T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
    operation: std::any::TypeId,
    expr: &'static str,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let Some((meta, len)) = binary_strided_meta(&a, &b, &out)? else {
        return Ok(());
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

/// Run `out[idx] = op(a[idx], b[idx])` over logical indices of `out_layout`.
pub fn binary_elementwise_strided_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    binary_elementwise_strided_into_expression::<T, N>(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<CudaC>>::EXPR,
    )
}

/// Run a scalar-aware binary operation over logical indices of `out_layout`.
pub fn binary_elementwise_strided_typed_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: TypedBinaryExpr<CudaC, T>,
    T: DialectScalar<CudaC> + Pod,
{
    binary_elementwise_strided_into_expression::<T, N>(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<CudaC, T>>::EXPR,
    )
}

/// Run `out = op(a, b)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// Inputs are broadcast to `output_shape` through the same layout contract as
/// [`binary_elementwise_strided_into`].
pub fn binary_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(len)?;
    binary_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        b,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}

/// Run a scalar-aware binary operation over `output_shape`, allocating a
/// contiguous output buffer.
pub fn binary_elementwise_strided_typed<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: TypedBinaryExpr<CudaC, T>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(len)?;
    binary_elementwise_strided_typed_into::<Op, T, N>(
        device,
        a,
        b,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}

/// Run `out[idx] = op(a[idx])` over logical indices of `out_layout`.
pub fn unary_elementwise_strided_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let Some((meta, len)) = unary_strided_meta(&a, &out)? else {
        return Ok(());
    };

    launch_unary_strided::<Op, T>(device, a.buffer, out.buffer, meta, width, len)
}

/// Run `out = op(a)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// The input is broadcast to `output_shape` through the same layout contract as
/// [`unary_elementwise_strided_into`].
pub fn unary_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(len)?;
    unary_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}
