use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, DialectScalar, ElementwiseOps, HephaestusError, Result, StridedView,
    TypedBinaryExpr, UnaryExpr, Wgsl,
};

use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::{
    checked_bind_group, checked_submit, device_owner, validate_device_owner,
};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, validate_out,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

/// Provider-owned implementation of [`ElementwiseOps`] over wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuElementwiseOps;

/// Prepared resources for one elementwise dispatch bound to fixed operands.
pub struct PreparedElementwise {
    owner: PipelineCache,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    groups: u32,
    _meta_buffer: Option<crate::infrastructure::pool::UniformBufferGuard>,
    _scalar_buffer: Option<crate::infrastructure::pool::UniformBufferGuard>,
}

// ── Kernel discriminators (separate pipeline-cache keys) ──────────────

struct SeamUnaryKernel<Op>(PhantomData<Op>);
struct SeamBinaryKernel<Op>(PhantomData<Op>);
struct SeamScalarKernel<Op>(PhantomData<Op>);

// ── Shader sources (mirror the strided module; TypeId keeps pipelines separate) ─

fn wgsl_meta() -> &'static str {
    "struct Meta {
        shape: vec4<u32>,
        a_strides: vec4<i32>,
        b_strides: vec4<i32>,
        out_strides: vec4<i32>,
        offsets: vec4<u32>,
    }"
}

fn wgsl_decode() -> &'static str {
    "    var rem = i;
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
"
}

fn unary_shader<T: DialectScalar<Wgsl>, Op: UnaryExpr<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{ return; }}
{decode}    let x = a[u32(a_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = wgsl_meta(),
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = wgsl_decode(),
        expr = <Op as UnaryExpr<Wgsl>>::EXPR,
    )
}

fn binary_shader<T: DialectScalar<Wgsl>>(width: BlockWidth, expr: &str) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{ return; }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = b[u32(b_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = wgsl_meta(),
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = wgsl_decode(),
        expr = expr,
    )
}

// ── Helpers ──────────────────────────────────────────────────────────

fn validate_rank<const N: usize>() -> Result<()> {
    if N > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "elementwise dispatch supports rank <= {MAX_STRIDED_RANK}, got rank {N}"
            ),
        });
    }
    Ok(())
}

/// Shared meta-building and bind-group creation for unary, binary, and typed
/// binary elementwise dispatches.
fn prepare_unary_inner<Op, T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
) -> Result<PreparedElementwise>
where
    Op: UnaryExpr<Wgsl> + 'static,
    T: DialectScalar<Wgsl> + Pod + 'static,
{
    validate_rank::<N>()?;

    let out_layout = output.layout;
    let a_layout = input
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_out(output.buffer, out_layout)?;
    if len == 0 {
        return Ok(PreparedElementwise {
            owner: device_owner(device),
            pipeline: None,
            bind_group: None,
            groups: 0,
            _meta_buffer: None,
            _scalar_buffer: None,
        });
    }

    let meta = crate::application::strided::StridedMeta {
        shape: pad_shape(out_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_strides: [0; 4],
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            crate::application::strided::to_u32(a_layout.offset(), "input offset")?,
            0,
            crate::application::strided::to_u32(out_layout.offset(), "output offset")?,
            crate::application::strided::to_u32(len, "dispatch size")?,
        ],
    };

    let groups = workgroups(len, BlockWidth::DEFAULT)?;
    let pipeline = try_cached_pipeline(
        device,
        (
            TypeId::of::<SeamUnaryKernel<Op>>(),
            TypeId::of::<T>(),
            BlockWidth::DEFAULT.get(),
        ),
        "hephaestus-seam-unary",
        || unary_shader::<T, Op>(BlockWidth::DEFAULT),
    )?;

    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<
        crate::application::strided::StridedMeta,
    >(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));

    let bind_group = checked_bind_group(
        device,
        &pipeline,
        "hephaestus-seam-unary",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
        ],
    )?;

    Ok(PreparedElementwise {
        owner: device_owner(device),
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups,
        _meta_buffer: Some(meta_buffer),
        _scalar_buffer: None,
    })
}

fn scalar_shader<T: DialectScalar<Wgsl>, Op: BinaryExpr<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<uniform> scalar: {ty};
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{ return; }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = scalar;
    out[u32(o_off)] = {expr};
}}
"#,
        meta = wgsl_meta(),
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = wgsl_decode(),
        expr = <Op as BinaryExpr<Wgsl>>::EXPR,
    )
}

/// Meta-building and bind-group creation for broadcast-scalar dispatches.
fn prepare_scalar_inner<Op, T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    scalar: T,
    output: StridedView<'_, WgpuBuffer<T>, N>,
) -> Result<PreparedElementwise>
where
    Op: BinaryExpr<Wgsl> + 'static,
    T: DialectScalar<Wgsl> + Pod + 'static,
{
    validate_rank::<N>()?;

    let out_layout = output.layout;
    let a_layout = input
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_out(output.buffer, out_layout)?;
    if len == 0 {
        return Ok(PreparedElementwise {
            owner: device_owner(device),
            pipeline: None,
            bind_group: None,
            groups: 0,
            _meta_buffer: None,
            _scalar_buffer: None,
        });
    }

    let meta = crate::application::strided::StridedMeta {
        shape: pad_shape(out_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_strides: [0; 4],
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            crate::application::strided::to_u32(a_layout.offset(), "input offset")?,
            0,
            crate::application::strided::to_u32(out_layout.offset(), "output offset")?,
            crate::application::strided::to_u32(len, "dispatch size")?,
        ],
    };

    let groups = workgroups(len, BlockWidth::DEFAULT)?;
    let pipeline = try_cached_pipeline(
        device,
        (
            TypeId::of::<SeamScalarKernel<Op>>(),
            TypeId::of::<T>(),
            BlockWidth::DEFAULT.get(),
        ),
        "hephaestus-seam-scalar",
        || scalar_shader::<T, Op>(BlockWidth::DEFAULT),
    )?;

    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<
        crate::application::strided::StridedMeta,
    >(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));

    let raw_scalar = device.get_uniform_buffer(WgpuDevice::byte_size::<T>(1)?)?;
    let scalar_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_scalar);
    device
        .queue()
        .write_buffer(&scalar_buffer, 0, bytemuck::bytes_of(&scalar));

    let bind_group = checked_bind_group(
        device,
        &pipeline,
        "hephaestus-seam-scalar",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scalar_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buffer.as_entire_binding(),
            },
        ],
    )?;

    Ok(PreparedElementwise {
        owner: device_owner(device),
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups,
        _meta_buffer: Some(meta_buffer),
        _scalar_buffer: Some(scalar_buffer),
    })
}

fn prepare_binary_inner<Op, T, const N: usize>(
    device: &WgpuDevice,
    lhs: StridedView<'_, WgpuBuffer<T>, N>,
    rhs: StridedView<'_, WgpuBuffer<T>, N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    expr: &'static str,
) -> Result<PreparedElementwise>
where
    Op: 'static,
    T: DialectScalar<Wgsl> + Pod + 'static,
{
    validate_rank::<N>()?;

    let out_layout = output.layout;
    let a_layout = lhs
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    let b_layout = rhs
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(lhs.buffer.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(rhs.buffer.len)
        .map_err(map_layout_err)?;
    if output.buffer.aliases(lhs.buffer) || output.buffer.aliases(rhs.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias any input buffer".to_string(),
        });
    }
    let len = validate_out(output.buffer, out_layout)?;
    if len == 0 {
        return Ok(PreparedElementwise {
            owner: device_owner(device),
            pipeline: None,
            bind_group: None,
            groups: 0,
            _meta_buffer: None,
            _scalar_buffer: None,
        });
    }

    let meta = crate::application::strided::StridedMeta {
        shape: pad_shape(out_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_strides: pad_strides(b_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            crate::application::strided::to_u32(a_layout.offset(), "input offset")?,
            crate::application::strided::to_u32(b_layout.offset(), "input offset")?,
            crate::application::strided::to_u32(out_layout.offset(), "output offset")?,
            crate::application::strided::to_u32(len, "dispatch size")?,
        ],
    };

    let groups = workgroups(len, BlockWidth::DEFAULT)?;
    let operation = TypeId::of::<SeamBinaryKernel<Op>>();
    let pipeline = try_cached_pipeline(
        device,
        (operation, TypeId::of::<T>(), BlockWidth::DEFAULT.get()),
        "hephaestus-seam-binary",
        || binary_shader::<T>(BlockWidth::DEFAULT, expr),
    )?;

    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<
        crate::application::strided::StridedMeta,
    >(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));

    let bind_group = checked_bind_group(
        device,
        &pipeline,
        "hephaestus-seam-binary",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lhs.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rhs.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buffer.as_entire_binding(),
            },
        ],
    )?;

    Ok(PreparedElementwise {
        owner: device_owner(device),
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups,
        _meta_buffer: Some(meta_buffer),
        _scalar_buffer: None,
    })
}

fn dispatch_prepared<const N: usize>(
    device: &WgpuDevice,
    prepared: &PreparedElementwise,
    label: &'static str,
) -> Result<()> {
    validate_device_owner(&prepared.owner, device, "elementwise operation")?;
    let Some(pipeline) = &prepared.pipeline else {
        return Ok(());
    };
    let Some(bind_group) = &prepared.bind_group else {
        return Ok(());
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    crate::application::pipeline::encode_compute_pass(
        &mut encoder,
        pipeline,
        bind_group,
        prepared.groups,
        label,
    );
    checked_submit(device, label, encoder)
}

// ── Trait implementation ─────────────────────────────────────────────

impl<T> ElementwiseOps<WgpuDevice, T> for WgpuElementwiseOps
where
    T: DialectScalar<Wgsl> + Pod + Send + Sync + 'static,
{
    type Dialect = Wgsl;
    type PreparedUnary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;
    type PreparedBinary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;
    type PreparedScalar<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;
    type PreparedTypedBinary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<T>, N>,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        prepare_unary_inner::<Op, T, N>(device, input, output)
    }

    fn dispatch_unary<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-unary")
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'op, WgpuBuffer<T>, N>,
        rhs: StridedView<'op, WgpuBuffer<T>, N>,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_binary_inner::<Op, T, N>(device, lhs, rhs, output, <Op as BinaryExpr<Wgsl>>::EXPR)
    }

    fn dispatch_binary<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-binary")
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<T>, N>,
        scalar: T,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_scalar_inner::<Op, T, N>(device, input, scalar, output)
    }

    fn dispatch_scalar<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-scalar")
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'op, WgpuBuffer<T>, N>,
        rhs: StridedView<'op, WgpuBuffer<T>, N>,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
        T: DialectScalar<Self::Dialect>,
    {
        prepare_binary_inner::<SeamTypedExpr<Op, T>, T, N>(
            device,
            lhs,
            rhs,
            output,
            <Op as TypedBinaryExpr<Wgsl, T>>::EXPR,
        )
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-typed-binary")
    }
}

/// Adapter so a `TypedBinaryExpr` can be used as a plain `BinaryExpr`
/// (the expression string is already dialect-specific, and the kernel
/// binds no additional scalar parameters).
#[derive(Clone, Copy)]
struct SeamTypedExpr<Op, T: Send + Sync + 'static>(PhantomData<(Op, T)>);

impl<
    Op: TypedBinaryExpr<Wgsl, T> + Send + Sync + 'static,
    T: DialectScalar<Wgsl> + Send + Sync + 'static,
> BinaryExpr<Wgsl> for SeamTypedExpr<Op, T>
{
    const EXPR: &'static str = <Op as TypedBinaryExpr<Wgsl, T>>::EXPR;
}
