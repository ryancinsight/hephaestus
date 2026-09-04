//! Monomorphized WGSL stateful parameter updates.

use core::marker::PhantomData;

use hephaestus_core::{
    BlockWidth, StatefulUpdateAliasing, StatefulUpdateMeta, StatefulUpdateOperands,
    StatefulUpdateOps, StatefulUpdateRule, Wgsl, plan_stateful_update,
};

use crate::application::elementwise::encode_elementwise;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::prepared::validate_buffer_owner;
use crate::{Result, WgpuBuffer, WgpuDevice};

struct StatefulKernel<Rule>(PhantomData<Rule>);

fn parameters_declaration<Rule>() -> String
where
    Rule: StatefulUpdateRule<Wgsl>,
{
    let fields = Rule::PARAMETER_FIELDS
        .iter()
        .map(|field| format!("    {field}: f32,\n"))
        .collect::<String>();
    format!("struct Parameters {{\n{fields}}}\n")
}

fn shader_source<Rule>(width: BlockWidth) -> String
where
    Rule: StatefulUpdateRule<Wgsl>,
{
    let state_one_binding = if Rule::STATE_COUNT == 2 {
        "@group(0) @binding(5) var<storage, read_write> state_one: array<f32>;\n"
    } else {
        ""
    };
    let state_one_decode = if Rule::STATE_COUNT == 2 {
        "    var state_one_offset = i32(lmeta.offsets.w);\n"
    } else {
        ""
    };
    let state_one_stride = if Rule::STATE_COUNT == 2 {
        "        state_one_offset = state_one_offset + index * lmeta.strides[3][lane][component];\n"
    } else {
        ""
    };
    let state_one_load = if Rule::STATE_COUNT == 2 {
        "    let state_one_value = state_one[u32(state_one_offset)];\n    var state_one_next = state_one_value;\n"
    } else {
        ""
    };
    let state_one_store = if Rule::STATE_COUNT == 2 {
        "    state_one[u32(state_one_offset)] = state_one_next;\n"
    } else {
        ""
    };
    format!(
        r#"struct Meta {{
    shape: array<vec4<u32>, 2>,
    strides: array<array<vec4<i32>, 2>, 4>,
    offsets: vec4<u32>,
    dispatch: vec4<u32>,
}}
{parameters}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@group(0) @binding(2) var<storage, read_write> parameter: array<f32>;
@group(0) @binding(3) var<storage, read> gradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> state_zero: array<f32>;
{state_one_binding}
@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let linear = gid.x;
    if (linear >= lmeta.dispatch.x) {{ return; }}
    var remainder = linear;
    var parameter_offset = i32(lmeta.offsets.x);
    var gradient_offset = i32(lmeta.offsets.y);
    var state_zero_offset = i32(lmeta.offsets.z);
{state_one_decode}    for (var dimension: i32 = 7; dimension >= 0; dimension = dimension - 1) {{
        let lane = dimension / 4;
        let component = dimension % 4;
        let extent = lmeta.shape[lane][component];
        let index = i32(remainder % extent);
        remainder = remainder / extent;
        parameter_offset = parameter_offset + index * lmeta.strides[0][lane][component];
        gradient_offset = gradient_offset + index * lmeta.strides[1][lane][component];
        state_zero_offset = state_zero_offset + index * lmeta.strides[2][lane][component];
{state_one_stride}    }}
    let parameter_value = parameter[u32(parameter_offset)];
    let gradient_value = gradient[u32(gradient_offset)];
    let state_zero_value = state_zero[u32(state_zero_offset)];
    var parameter_next = parameter_value;
    var state_zero_next = state_zero_value;
{state_one_load}    {body}
    parameter[u32(parameter_offset)] = parameter_next;
    state_zero[u32(state_zero_offset)] = state_zero_next;
{state_one_store}}}
"#,
        parameters = parameters_declaration::<Rule>(),
        width = width.get(),
        body = Rule::BODY,
    )
}

fn aliasing<const N: usize>(
    operands: &StatefulUpdateOperands<'_, WgpuBuffer<f32>, N>,
) -> StatefulUpdateAliasing {
    let state_zero = operands.states.first();
    let state_one = operands.states.get(1);
    StatefulUpdateAliasing {
        parameter_gradient: operands.parameter.buffer.aliases(operands.gradient.buffer),
        parameter_state_zero: state_zero
            .is_some_and(|state| operands.parameter.buffer.aliases(state.buffer)),
        parameter_state_one: state_one
            .is_some_and(|state| operands.parameter.buffer.aliases(state.buffer)),
        gradient_state_zero: state_zero
            .is_some_and(|state| operands.gradient.buffer.aliases(state.buffer)),
        gradient_state_one: state_one
            .is_some_and(|state| operands.gradient.buffer.aliases(state.buffer)),
        states: state_zero
            .zip(state_one)
            .is_some_and(|(left, right)| left.buffer.aliases(right.buffer)),
    }
}

/// Provider-owned WGPU implementation of stateful parameter updates.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuStatefulUpdateOps;

impl StatefulUpdateOps<WgpuDevice> for WgpuStatefulUpdateOps {
    type Dialect = Wgsl;

    fn stateful_update<Rule, const N: usize>(
        &self,
        device: &WgpuDevice,
        operands: StatefulUpdateOperands<'_, WgpuBuffer<f32>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        Rule::validate_parameters(&parameters)?;
        validate_buffer_owner(
            operands.parameter.buffer,
            device,
            "stateful-update parameter",
        )?;
        validate_buffer_owner(operands.gradient.buffer, device, "stateful-update gradient")?;
        for state in operands.states {
            validate_buffer_owner(state.buffer, device, "stateful-update state")?;
        }
        let plan = plan_stateful_update(operands, Rule::STATE_COUNT, aliasing(&operands))?;
        if plan.is_empty() {
            return Ok(());
        }
        let width = BlockWidth::DEFAULT;
        let key = (
            core::any::TypeId::of::<StatefulKernel<Rule>>(),
            core::any::TypeId::of::<f32>(),
            width.get(),
        );
        let pipeline = cached_pipeline(device, key, "hephaestus-stateful-update", || {
            shader_source::<Rule>(width)
        });
        let raw_meta =
            device.get_uniform_buffer(WgpuDevice::byte_size::<StatefulUpdateMeta>(1)?)?;
        let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
        let raw_parameters =
            device.get_uniform_buffer(WgpuDevice::byte_size::<Rule::Parameters>(1)?)?;
        let parameter_buffer =
            crate::infrastructure::pool::uniform_guard(device.clone(), raw_parameters);
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&plan.metadata()));
        device
            .queue()
            .write_buffer(&parameter_buffer, 0, eunomia::layout::bytes_of(&parameters));
        let state_zero = operands
            .states
            .first()
            .expect("invariant: planner validated state zero");
        let common = [
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
                resource: operands.parameter.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: operands.gradient.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state_zero.buffer.as_entire_binding(),
            },
        ];
        if let Some(state_one) = operands.states.get(1) {
            let entries = [
                common[0].clone(),
                common[1].clone(),
                common[2].clone(),
                common[3].clone(),
                common[4].clone(),
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state_one.buffer.as_entire_binding(),
                },
            ];
            encode_elementwise(
                device,
                &pipeline,
                "hephaestus-stateful-update",
                &entries,
                workgroups(plan.len(), width)?,
            )
        } else {
            encode_elementwise(
                device,
                &pipeline,
                "hephaestus-stateful-update",
                &common,
                workgroups(plan.len(), width)?,
            )
        }
    }
}
