use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, DialectScalar, MAX_STRIDED_RANK, Result, StridedMeta,
    Wgsl, map_layout_err, pad_shape, pad_strides, to_u32,
};
use leto::Layout;

use crate::application::canary;
use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::encode::encode_strided;
use super::shaders::scalar_shader;
use super::types::{StridedOperand, StridedScalarKernel};
use super::validate::validate_out;

/// Run `out[idx] = op(a[idx], scalar)` over logical indices of `out_layout`,
/// with `a` broadcast to the output shape by leto's broadcast rules.
///
/// The scalar is supplied through a small pooled `uniform` (the same mechanism
/// as the contiguous [`scalar_elementwise_into`](crate::application::elementwise::scalar)
/// path), so no per-call device storage buffer is allocated and uploaded for the
/// one-element operand. Scalar semantics stay identical to `op(a, scalar)`; the
/// dedicated kernel reuses the shared strided metadata/decode/encode core.
pub fn scalar_elementwise_strided_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 8");
    }
    canary::require_template_safe::<T>(device, Op::EXPR, "strided scalar elementwise")?;

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
        // `b` operand is unused by the scalar kernel; zeroed for the shared decode.
        b_strides: [0i32; 8],
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(a_layout.offset(), "input offset")?,
            0,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    // Pooled uniform scalar (matches the contiguous scalar path): no per-call
    // storage operand allocation. queue.write_buffer is queue-ordered, so the
    // recycled uniform cannot race in-flight dispatches.
    let scalar_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<T>(1)?)?;

    device
        .queue()
        .write_buffer(&scalar_buffer, 0, eunomia::layout::bytes_of(&scalar));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<StridedScalarKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-strided-scalar",
        || scalar_shader::<Op, T>(width),
    );
    encode_strided(
        device,
        &pipeline,
        &meta,
        &[&a.buffer.buffer, &scalar_buffer, &out.buffer.buffer],
        len,
        width,
        "hephaestus-strided-scalar",
    )
}

/// Run `out = op(a, scalar)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// Delegates to [`scalar_elementwise_strided_into`] over a freshly allocated
/// dense output.
pub fn scalar_elementwise_strided<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(len)?;
    scalar_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        scalar,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}
