//! Runtime-parameter unary elementwise dispatch over WGPU.

use core::marker::PhantomData;

use hephaestus_core::{
    BlockWidth, HephaestusError, ParameterizedUnaryExpr, ParameterizedUnaryOps, Result,
    StridedView, Wgsl, validate_parameterized_output,
};

use crate::application::elementwise::encode_elementwise;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::strided::{
    StridedMeta, StridedOperand, WGSL_DECODE, WGSL_META, map_layout_err, pad_shape, pad_strides,
    to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

struct ParameterizedUnaryKernel<Op>(PhantomData<Op>);

fn shader_source<Op>(width: BlockWidth) -> String
where
    Op: ParameterizedUnaryExpr<Wgsl>,
{
    format!(
        r#"{meta}
struct Parameters {{
    first: {ty},
    second: {ty},
}}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@group(0) @binding(2) var<storage, read> input: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> output: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{ return; }}
{decode}    let x = input[u32(a_off)];
    let first = parameters.first;
    let second = parameters.second;
    output[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = "f32",
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = Op::EXPR,
    )
}

/// Run a runtime-parameter unary expression over rank-`N` strided views.
///
/// # Errors
///
/// Returns a layout, shape, alias, workgroup, pipeline, or submission error.
pub fn parameterized_unary_strided_into<Op, const N: usize>(
    device: &WgpuDevice,
    input: StridedOperand<'_, f32, N>,
    parameters: [f32; 2],
    output: StridedOperand<'_, f32, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: ParameterizedUnaryExpr<Wgsl>,
{
    const {
        assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
    }
    let input_layout = input
        .layout
        .broadcast(output.layout.shape())
        .map_err(map_layout_err)?;
    input_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_parameterized_output(output.layout, output.buffer.len)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(output.layout.shape())?,
        a_strides: pad_strides(input_layout.strides())?,
        b_strides: [0; 4],
        out_strides: pad_strides(output.layout.strides())?,
        offsets: [
            to_u32(input_layout.offset(), "input offset")?,
            0,
            to_u32(output.layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };
    let key = (
        core::any::TypeId::of::<ParameterizedUnaryKernel<Op>>(),
        core::any::TypeId::of::<f32>(),
        width.get(),
    );
    let pipeline = cached_pipeline(device, key, "hephaestus-parameterized-unary", || {
        shader_source::<Op>(width)
    });

    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<StridedMeta>(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
    let raw_parameters = device.get_uniform_buffer(WgpuDevice::byte_size::<[f32; 2]>(1)?)?;
    let parameter_buffer =
        crate::infrastructure::pool::uniform_guard(device.clone(), raw_parameters);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));
    device
        .queue()
        .write_buffer(&parameter_buffer, 0, bytemuck::bytes_of(&parameters));

    encode_elementwise(
        device,
        &pipeline,
        "hephaestus-parameterized-unary",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: parameter_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buffer.as_entire_binding(),
            },
        ],
        workgroups(len, width)?,
    )
}

/// Provider-owned implementation of [`ParameterizedUnaryOps`] for WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuParameterizedUnaryOps;

impl ParameterizedUnaryOps<WgpuDevice> for WgpuParameterizedUnaryOps {
    type Dialect = Wgsl;

    fn parameterized_unary_into<Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<f32>, N>,
        parameters: [f32; 2],
        output: StridedView<'_, WgpuBuffer<f32>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>,
    {
        parameterized_unary_strided_into::<Op, N>(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
            parameters,
            StridedOperand {
                buffer: output.buffer,
                layout: output.layout,
            },
            BlockWidth::DEFAULT,
        )
    }
}

pub use hephaestus_core::{HardtanhGradOp, HardtanhOp, ThresholdGradOp, ThresholdOp};
