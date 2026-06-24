//! Strided-layout-aware dispatch over leto host-side layout metadata.
//!
//! Consumers describe operands with [`leto::Layout`] (shape/strides/offset) so
//! transposed, sliced, and broadcast (zero-stride) views dispatch directly —
//! no host-side materialization into contiguous staging copies. Inputs
//! broadcast to the output shape with leto's own broadcast rules, keeping
//! device semantics identical to leto's CPU `binary_map`.
//!
//! One shared metadata/pipeline/encode core serves the binary, unary, and
//! scalar op families. The scalar family has a dedicated kernel that reads the
//! broadcast scalar from a small pooled `uniform` (like the contiguous
//! `scalar_elementwise_into` path) rather than uploading a one-element storage
//! operand per call, so a strided scalar op allocates no per-call device buffer.

use core::marker::PhantomData;
use std::any::TypeId;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};
use leto::Layout;

use crate::application::elementwise::{BinaryWgslOp, UnaryWgslOp};
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Maximum rank the packed rank-4 metadata covers. Lower-rank layouts are
/// padded with leading size-1 / stride-0 dimensions, which contribute nothing
/// to the offset computation.
pub const MAX_STRIDED_RANK: usize = 4;

/// A device buffer paired with the leto layout describing its logical view:
/// the unit every strided operand is passed as. Plain `Copy` references —
/// bundling is purely to keep signatures at parameter-object altitude.
#[derive(Clone, Copy)]
pub struct StridedOperand<'a, T, const N: usize> {
    /// The device buffer.
    pub buffer: &'a WgpuBuffer<T>,
    /// The logical layout over that buffer.
    pub layout: &'a Layout<N>,
}

/// Pipeline-cache discriminators so strided kernels never collide with the
/// contiguous kernels of the same `Op` in the `(TypeId, TypeId)` cache key.
struct StridedBinaryKernel<Op>(PhantomData<Op>);
struct StridedUnaryKernel<Op>(PhantomData<Op>);
struct StridedScalarKernel<Op>(PhantomData<Op>);

/// Packed layout metadata matching the WGSL `Meta` uniform: rank-4 padded
/// shape, per-operand strides, and `[a_off, b_off, out_off, len]`. The unary
/// family reuses the same struct with the `b` lanes zeroed so one packing
/// path and one uniform layout serve every strided kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct StridedMeta {
    pub(crate) shape: [u32; 4],
    pub(crate) a_strides: [i32; 4],
    pub(crate) b_strides: [i32; 4],
    pub(crate) out_strides: [i32; 4],
    pub(crate) offsets: [u32; 4],
}

/// WGSL `Meta` declaration shared by every strided kernel.
pub(crate) const WGSL_META: &str = r"struct Meta {
    shape: vec4<u32>,
    a_strides: vec4<i32>,
    b_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
}
";

/// Flat-index → per-operand-offset decode shared by every strided kernel.
/// `arrayLength` cannot guard strided access, so the logical length travels
/// in `offsets.w` and is checked by each kernel before this body runs.
pub(crate) const WGSL_DECODE: &str = r"    var rem = i;
    var a_off = i32(lmeta.offsets.x);
    var b_off = i32(lmeta.offsets.y);
    var o_off = i32(lmeta.offsets.z);
    for (var d: i32 = 3; d >= 0; d = d - 1) {
        let dim = lmeta.shape[d];
        let idx = i32(rem % dim);
        rem = rem / dim;
        a_off = a_off + idx * lmeta.a_strides[d];
        b_off = b_off + idx * lmeta.b_strides[d];
        o_off = o_off + idx * lmeta.out_strides[d];
    }
";

#[inline]
pub(crate) fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

#[inline]
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

#[inline]
pub(crate) fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

#[inline]
pub(crate) fn pad_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 4]> {
    let mut out = [1u32; 4];
    for (d, &dim) in shape.iter().enumerate() {
        out[4 - N + d] = to_u32(dim, "dimension")?;
    }
    Ok(out)
}

#[inline]
pub(crate) fn pad_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 4]> {
    let mut out = [0i32; 4];
    for (d, &stride) in strides.iter().enumerate() {
        out[4 - N + d] = to_i32(stride, "stride")?;
    }
    Ok(out)
}

/// Validate an output layout against its buffer and return the logical length.
fn validate_out<T, const N: usize>(out: &WgpuBuffer<T>, out_layout: &Layout<N>) -> Result<usize> {
    if out_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    out_layout
        .validate_storage_len(out.len)
        .map_err(map_layout_err)?;
    out_layout.checked_size().map_err(map_layout_err)
}

/// Upload the meta uniform, bind `buffers` after it at consecutive slots, and
/// dispatch `len` invocations: the single encode path shared by every strided
/// kernel.
fn encode_strided(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    meta: &StridedMeta,
    buffers: &[&wgpu::Buffer],
    len: usize,
    width: BlockWidth,
    label: &str,
) -> Result<()> {
    let groups = workgroups(len, width)?;

    // Pooled meta uniform: queue.write_buffer is ordered on the queue
    // timeline, so recycling after submit cannot race in-flight dispatches.
    let raw_meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<StridedMeta>(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(meta));

    let mut entries = Vec::with_capacity(buffers.len() + 1);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: meta_buffer.as_entire_binding(),
    });
    for (slot, buffer) in buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: u32::try_from(slot + 1).expect("invariant: strided bind slot count fits u32"),
            resource: buffer.as_entire_binding(),
        });
    }
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

fn binary_shader<Op: BinaryWgslOp, T: WgslScalar>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{
        return;
    }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = b[u32(b_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = Op::WGSL_EXPR,
    )
}

fn unary_shader<Op: UnaryWgslOp, T: WgslScalar>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{
        return;
    }}
{decode}    let x = a[u32(a_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = Op::WGSL_EXPR,
    )
}

/// Strided scalar kernel: one strided storage operand `a`, the broadcast scalar
/// supplied through a small `uniform` (no per-call storage operand), strided
/// output. Mirrors [`scalar_elementwise_into`](super::elementwise::scalar) but
/// over leto strided layouts. The shared `WGSL_DECODE` computes an unused
/// `b_off` here (identical to the unary kernel), which the WGSL compiler elides.
fn scalar_shader<Op: BinaryWgslOp, T: WgslScalar>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<uniform> scalar: {ty};
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{
        return;
    }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = scalar;
    out[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = Op::WGSL_EXPR,
    )
}

/// Run `out[idx] = op(a[idx], b[idx])` over logical indices of `out_layout`,
/// with `a`/`b` broadcast to the output shape by leto's broadcast rules.
///
/// All three operands are described by leto host-side layouts; transposed,
/// sliced, offset, and zero-stride (broadcast) inputs dispatch without any
/// contiguous materialization. The output buffer is caller-owned (allocation
/// control stays with the consumer); zero-stride aliasing in the output
/// layout is rejected because concurrent invocations would race on one
/// physical element. Rank is capped at [`MAX_STRIDED_RANK`] at compile time.
pub fn binary_elementwise_strided_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    let b_layout = b
        .layout
        .broadcast(out_layout.shape)
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
        shape: pad_shape(out_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        b_strides: pad_strides(b_layout.strides)?,
        out_strides: pad_strides(out_layout.strides)?,
        offsets: [
            to_u32(a_layout.offset, "input offset")?,
            to_u32(b_layout.offset, "input offset")?,
            to_u32(out_layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };
    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<StridedBinaryKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-strided-binary",
        || binary_shader::<Op, T>(width),
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
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(len)?;
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

/// Run `out[idx] = op(a[idx])` over logical indices of `out_layout`, with `a`
/// broadcast to the output shape. Same layout semantics, validation, and
/// caller-owned output contract as [`binary_elementwise_strided_into`].
pub fn unary_elementwise_strided_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryWgslOp,
    T: WgslScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        b_strides: [0; 4],
        out_strides: pad_strides(out_layout.strides)?,
        offsets: [
            to_u32(a_layout.offset, "input offset")?,
            0,
            to_u32(out_layout.offset, "output offset")?,
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
    Op: UnaryWgslOp,
    T: WgslScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(len)?;
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

/// Run `out[idx] = op(a[idx], scalar)` over logical indices of `out_layout`,
/// with `a` broadcast to the output shape by leto's broadcast rules.
///
/// The scalar is supplied through a small pooled `uniform` (the same mechanism
/// as the contiguous [`scalar_elementwise_into`](super::elementwise::scalar)
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
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        // `b` operand is unused by the scalar kernel; zeroed for the shared decode.
        b_strides: [0i32; 4],
        out_strides: pad_strides(out_layout.strides)?,
        offsets: [
            to_u32(a_layout.offset, "input offset")?,
            0,
            to_u32(out_layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    // Pooled uniform scalar (matches the contiguous scalar path): no per-call
    // storage operand allocation. queue.write_buffer is queue-ordered, so the
    // recycled uniform cannot race in-flight dispatches.
    let raw_scalar_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<T>(1)?)?;
    let scalar_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_scalar_buf);
    device
        .queue()
        .write_buffer(&scalar_buffer, 0, bytemuck::bytes_of(&scalar));

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
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(len)?;
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
