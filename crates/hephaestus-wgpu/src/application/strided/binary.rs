use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, DialectScalar, MAX_STRIDED_RANK, Result, StridedMeta,
    TypedBinaryExpr, Wgsl, map_layout_err, pad_shape, pad_strides, to_u32,
};
use leto::Layout;

use crate::application::canary;
use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::encode::encode_strided;
use super::shaders::binary_shader;
use super::types::{StridedBinaryKernel, StridedOperand};
use super::validate::validate_out;

/// Run `out[idx] = op(a[idx], b[idx])` over logical indices of `out_layout`,
/// with `a`/`b` broadcast to the output shape by leto's broadcast rules.
///
/// All three operands are described by leto host-side layouts; transposed,
/// sliced, offset, and zero-stride (broadcast) inputs dispatch without any
/// contiguous materialization. The output buffer is caller-owned (allocation
/// control stays with the consumer); zero-stride aliasing in the output
/// layout is rejected because concurrent invocations would race on one
/// physical element. Rank is capped at [`MAX_STRIDED_RANK`] at compile time.
fn binary_elementwise_strided_into_expression<T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
    operation: TypeId,
    expr: &'static str,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 8");
    }
    canary::require_template_safe::<T>(device, expr, "strided binary elementwise")?;

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    let b_layout = b
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.buffer.len)
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_strides: pad_strides(b_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(a_layout.offset(), "input offset")?,
            to_u32(b_layout.offset(), "input offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };
    let pipeline = cached_pipeline(
        device,
        (operation, TypeId::of::<T>(), width.get()),
        "hephaestus-strided-binary",
        || binary_shader::<T>(width, expr),
    );
    encode_strided(
        device,
        &pipeline,
        &meta,
        &[&a.buffer.buffer, &b.buffer.buffer, &out.buffer.buffer],
        len,
        width,
        "hephaestus-strided-binary",
    )
}

/// Run a scalar-aware binary operation over broadcastable strided operands.
pub fn binary_elementwise_strided_typed_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: TypedBinaryExpr<Wgsl, T>,
    T: DialectScalar<Wgsl> + Pod,
{
    binary_elementwise_strided_into_expression::<T, N>(
        device,
        a,
        b,
        out,
        width,
        TypeId::of::<StridedBinaryKernel<Op>>(),
        <Op as TypedBinaryExpr<Wgsl, T>>::EXPR,
    )
}

/// Run a scalar-aware binary operation over strided operands, allocating a
/// contiguous output buffer.
pub fn binary_elementwise_strided_typed<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: TypedBinaryExpr<Wgsl, T>,
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 8");
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

/// Run `out[idx] = op(a[idx], b[idx])` over logical indices of `out_layout`.
pub fn binary_elementwise_strided_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    binary_elementwise_strided_into_expression::<T, N>(
        device,
        a,
        b,
        out,
        width,
        TypeId::of::<StridedBinaryKernel<Op>>(),
        <Op as BinaryExpr<Wgsl>>::EXPR,
    )
}

/// Run `out = op(a, b)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// Inputs are broadcast to `output_shape` through the same layout contract as
/// [`binary_elementwise_strided_into`].
pub fn binary_elementwise_strided<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 8");
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
