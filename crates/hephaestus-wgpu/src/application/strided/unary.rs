use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, MAX_STRIDED_RANK, Result, StridedMeta, UnaryExpr,
    Wgsl, map_layout_err, pad_shape, pad_strides, to_u32,
};
use leto::Layout;

use crate::application::canary;
use crate::application::elementwise::reject_unsupported_unary;
use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::encode::encode_strided;
use super::shaders::unary_shader;
use super::types::{StridedOperand, StridedUnaryKernel};
use super::validate::validate_out;

/// Run `out[idx] = op(a[idx])` over logical indices of `out_layout`, with `a`
/// broadcast to the output shape. Same layout semantics, validation, and
/// caller-owned output contract as [`binary_elementwise_strided_into`](super::binary::binary_elementwise_strided_into).
pub fn unary_elementwise_strided_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 8");
    }
    reject_unsupported_unary::<Op, T>()?;
    canary::require_template_safe::<T>(device, Op::EXPR, "strided unary elementwise")?;

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_strides: [0; 8],
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(a_layout.offset(), "input offset")?,
            0,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };
    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<StridedUnaryKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-strided-unary",
        || unary_shader::<Op, T>(width),
    );
    encode_strided(
        device,
        &pipeline,
        &meta,
        &[&a.buffer.buffer, &out.buffer.buffer],
        len,
        width,
        "hephaestus-strided-unary",
    )
}

/// Run `out = op(a)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// The input is broadcast to `output_shape` through the same layout contract as
/// [`unary_elementwise_strided_into`].
pub fn unary_elementwise_strided<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 8");
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
