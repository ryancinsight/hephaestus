use std::any::TypeId;
use std::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::{
    AxisScanMeta, BlockWidth, CombineExpr, DialectScalar, HephaestusError, IdentityToken,
    OpIdentity, Result, ScanDirection, ScanOps, StridedView, Wgsl, plan_axis_scan,
};

use crate::application::pipeline::try_cached_pipeline;
use crate::application::prepared::{
    checked_bind_group, checked_submit, device_owner, validate_device_owner,
};
use crate::application::strided::validate_out;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

/// Provider-owned implementation of [`ScanOps`] over wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuScanOps;

/// Prepared resources for one scan dispatch bound to fixed operands.
pub struct PreparedScan {
    owner: PipelineCache,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    groups: u32,
    _meta_buffer: Option<crate::infrastructure::pool::UniformBufferGuard>,
}

impl<T> ScanOps<WgpuDevice, T> for WgpuScanOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;
    type PreparedScan<'op, const N: usize>
        = PreparedScan
    where
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("scan currently supports only rank-2 operands, got rank {N}"),
            });
        }

        // SAFETY: N == 2 verified above; Layout<N> at runtime is Layout<2>.
        let input_layout: &leto::Layout<2> =
            unsafe { &*(input.layout as *const leto::Layout<N> as *const leto::Layout<2>) };
        let output_layout: &leto::Layout<2> =
            unsafe { &*(output.layout as *const leto::Layout<N> as *const leto::Layout<2>) };
        validate_out(output.buffer, output.layout)?;
        let Some(dispatch) = plan_axis_scan(
            input_layout,
            input.buffer.len,
            output_layout,
            output.buffer.len,
            axis,
            direction,
            BlockWidth::DEFAULT,
            input.buffer.aliases(output.buffer),
        )?
        else {
            return Ok(PreparedScan {
                owner: device_owner(device),
                pipeline: None,
                bind_group: None,
                groups: 0,
                _meta_buffer: None,
            });
        };

        let pipeline = try_cached_pipeline(
            device,
            (
                TypeId::of::<ScanSeamKernel<Op>>(),
                TypeId::of::<T>(),
                BlockWidth::DEFAULT.get(),
            ),
            "hephaestus-axis-scan-seam",
            || scan_shader_source::<Op, T>(BlockWidth::DEFAULT),
        )?;

        let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<AxisScanMeta>(1)?)?;
        let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&dispatch.meta));

        let bind_group = checked_bind_group(
            device,
            &pipeline,
            "hephaestus-axis-scan-seam",
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

        Ok(PreparedScan {
            owner: device_owner(device),
            pipeline: Some(pipeline),
            bind_group: Some(bind_group),
            groups: dispatch.groups,
            _meta_buffer: Some(meta_buffer),
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        validate_device_owner(&prepared.owner, device, "scan")?;
        let Some(pipeline) = &prepared.pipeline else {
            return Ok(());
        };
        let Some(bind_group) = &prepared.bind_group else {
            return Ok(());
        };

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-axis-scan"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-prepared-axis-scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(prepared.groups, 1, 1);
        }
        checked_submit(device, "hephaestus-prepared-axis-scan", encoder)
    }
}

struct ScanSeamKernel<Op>(PhantomData<Op>);

fn scan_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: CombineExpr<Wgsl>,
    T: IdentityToken<Op, Wgsl>,
{
    format!(
        r#"
struct AxisScanMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> scan_meta: AxisScanMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

fn source_offset(row: u32, col: u32) -> u32 {{
    let off = i32(scan_meta.offsets.x)
        + i32(row) * scan_meta.input_strides.x
        + i32(col) * scan_meta.input_strides.y;
    return u32(off);
}}

fn dest_offset(row: u32, col: u32) -> u32 {{
    let off = i32(scan_meta.offsets.y)
        + i32(row) * scan_meta.output_strides.x
        + i32(col) * scan_meta.output_strides.y;
    return u32(off);
}}

var<workgroup> partial: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {{
    let line = workgroup.x;
    let lane = local.x;

    let rows = scan_meta.input_shape.x;
    let cols = scan_meta.input_shape.y;
    let axis = scan_meta.offsets.z & 1u;
    let reverse = (scan_meta.offsets.z & 2u) != 0u;
    let len = select(cols, rows, axis == 0u);
    let chunk_len = (len + {wg}u - 1u) / {wg}u;
    let start = lane * chunk_len;
    let end = min(start + chunk_len, len);
    var local_acc: {ty} = {identity};

    for (var s = start; s < end; s = s + 1u) {{
        let idx = select(s, len - 1u - s, reverse);
        let row = select(line, idx, axis == 0u);
        let col = select(idx, line, axis == 0u);
        let lhs = local_acc;
        let rhs = input[source_offset(row, col)];
        local_acc = {expr};
        output[dest_offset(row, col)] = local_acc;
    }}
    partial[lane] = local_acc;
    workgroupBarrier();

    if (lane == 0u) {{
        var prefix: {ty} = {identity};
        for (var chunk = 0u; chunk < {wg}u; chunk = chunk + 1u) {{
            let total = partial[chunk];
            partial[chunk] = prefix;
            let lhs = prefix;
            let rhs = total;
            prefix = {expr};
        }}
    }}
    workgroupBarrier();

    let prefix = partial[lane];
    for (var s = start; s < end; s = s + 1u) {{
        let idx = select(s, len - 1u - s, reverse);
        let row = select(line, idx, axis == 0u);
        let col = select(idx, line, axis == 0u);
        let lhs = prefix;
        let rhs = output[dest_offset(row, col)];
        output[dest_offset(row, col)] = {expr};
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}
