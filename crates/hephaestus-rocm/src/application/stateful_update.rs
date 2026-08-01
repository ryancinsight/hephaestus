//! Provider-owned stateful parameter updates for ROCm.

use hephaestus_core::{
    BlockWidth, HipC, Result, StatefulUpdateAliasing, StatefulUpdateMeta, StatefulUpdateOperands,
    StatefulUpdateOps, StatefulUpdateRule, plan_stateful_update,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};

const ENTRY_POINT: &str = "stateful_update_kernel";

fn parameters_declaration<Rule>() -> String
where
    Rule: StatefulUpdateRule<HipC>,
{
    let fields = Rule::PARAMETER_FIELDS
        .iter()
        .map(|field| format!("    float {field};\n"))
        .collect::<String>();
    format!("typedef struct {{\n{fields}}} StatefulUpdateParameters;\n")
}

fn kernel_source<Rule>() -> String
where
    Rule: StatefulUpdateRule<HipC>,
{
    let state_one_argument = if Rule::STATE_COUNT == 2 {
        ",\n    float* state_one"
    } else {
        ""
    };
    let state_one_offset = if Rule::STATE_COUNT == 2 {
        "    long long state_one_offset = (long long)meta.offsets[3];\n"
    } else {
        ""
    };
    let state_one_decode = if Rule::STATE_COUNT == 2 {
        "        state_one_offset += (long long)coordinate * (long long)meta.strides[3][word][lane];\n"
    } else {
        ""
    };
    let state_one_load = if Rule::STATE_COUNT == 2 {
        "    float state_one_value = state_one[state_one_offset];\n    float state_one_next;\n"
    } else {
        ""
    };
    let state_one_store = if Rule::STATE_COUNT == 2 {
        "    state_one[state_one_offset] = state_one_next;\n"
    } else {
        ""
    };

    format!(
        r#"
typedef struct {{
    unsigned int shape[2][4];
    int strides[4][2][4];
    unsigned int offsets[4];
    unsigned int dispatch[4];
}} StatefulUpdateMeta;

{parameters}

extern "C" __global__ void {entry_point}(
    StatefulUpdateMeta meta,
    StatefulUpdateParameters parameters,
    float* parameter,
    const float* gradient,
    float* state_zero{state_one_argument}
) {{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= meta.dispatch[0]) {{
        return;
    }}

    unsigned int remaining = index;
    long long parameter_offset = (long long)meta.offsets[0];
    long long gradient_offset = (long long)meta.offsets[1];
    long long state_zero_offset = (long long)meta.offsets[2];
{state_one_offset}    for (int axis = 7; axis >= 0; --axis) {{
        unsigned int word = (unsigned int)axis / 4;
        unsigned int lane = (unsigned int)axis % 4;
        unsigned int coordinate = remaining % meta.shape[word][lane];
        remaining /= meta.shape[word][lane];
        parameter_offset += (long long)coordinate * (long long)meta.strides[0][word][lane];
        gradient_offset += (long long)coordinate * (long long)meta.strides[1][word][lane];
        state_zero_offset += (long long)coordinate * (long long)meta.strides[2][word][lane];
{state_one_decode}    }}

    float parameter_value = parameter[parameter_offset];
    float gradient_value = gradient[gradient_offset];
    float state_zero_value = state_zero[state_zero_offset];
    float parameter_next = parameter_value;
    float state_zero_next = state_zero_value;
{state_one_load}    {body}

    parameter[parameter_offset] = parameter_next;
    state_zero[state_zero_offset] = state_zero_next;
{state_one_store}}}
"#,
        parameters = parameters_declaration::<Rule>(),
        entry_point = ENTRY_POINT,
        body = Rule::BODY,
    )
}

fn aliasing<const N: usize>(
    operands: &StatefulUpdateOperands<'_, RocmBuffer<f32>, N>,
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
            .is_some_and(|(zero, one)| zero.buffer.aliases(one.buffer)),
    }
}

fn launch<Rule, const N: usize>(
    device: &RocmDevice,
    operands: StatefulUpdateOperands<'_, RocmBuffer<f32>, N>,
    parameters: <Rule as StatefulUpdateRule<HipC>>::Parameters,
    meta: StatefulUpdateMeta,
    len: usize,
) -> Result<()>
where
    Rule: StatefulUpdateRule<HipC>,
{
    let width = BlockWidth::DEFAULT;
    let kernel = cached_kernel(
        device,
        PipelineKey::StatefulUpdate {
            rule: core::any::TypeId::of::<Rule>(),
            scalar: core::any::TypeId::of::<f32>(),
            width: width.get(),
        },
        ENTRY_POINT,
        kernel_source::<Rule>,
    )?;
    let mut meta = meta;
    let mut parameters = parameters;
    let mut parameter: DevicePtr = operands.parameter.buffer.raw();
    let mut gradient: DevicePtr = operands.gradient.buffer.raw();
    let mut state_zero: DevicePtr = operands
        .states
        .first()
        .expect("invariant: stateful-update planning accepted state zero")
        .buffer
        .raw();
    let config = LaunchConfig::linear(grid_size(len, width)?, width);

    if Rule::STATE_COUNT == 2 {
        let mut state_one: DevicePtr = operands
            .states
            .get(1)
            .expect("invariant: two-state rule passed stateful-update planning")
            .buffer
            .raw();
        let mut args: [*mut core::ffi::c_void; 6] = [
            (&mut meta as *mut StatefulUpdateMeta).cast(),
            (&mut parameters as *mut Rule::Parameters).cast(),
            (&mut parameter as *mut DevicePtr).cast(),
            (&mut gradient as *mut DevicePtr).cast(),
            (&mut state_zero as *mut DevicePtr).cast(),
            (&mut state_one as *mut DevicePtr).cast(),
        ];
        launch_kernel(device, &kernel, config, &mut args)
    } else {
        let mut args: [*mut core::ffi::c_void; 5] = [
            (&mut meta as *mut StatefulUpdateMeta).cast(),
            (&mut parameters as *mut Rule::Parameters).cast(),
            (&mut parameter as *mut DevicePtr).cast(),
            (&mut gradient as *mut DevicePtr).cast(),
            (&mut state_zero as *mut DevicePtr).cast(),
        ];
        launch_kernel(device, &kernel, config, &mut args)
    }
}

/// Provider-owned ROCm implementation of stateful parameter updates.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmStatefulUpdateOps;

impl StatefulUpdateOps<RocmDevice> for RocmStatefulUpdateOps {
    type Dialect = HipC;

    fn stateful_update<Rule, const N: usize>(
        &self,
        device: &RocmDevice,
        operands: StatefulUpdateOperands<'_, RocmBuffer<f32>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        Rule::validate_parameters(&parameters)?;
        let plan = plan_stateful_update(operands, Rule::STATE_COUNT, aliasing(&operands))?;
        if plan.is_empty() {
            return Ok(());
        }

        launch::<Rule, N>(device, operands, parameters, plan.metadata(), plan.len())
    }
}

#[cfg(test)]
mod tests {
    use hephaestus_core::{Adam, Sgd};

    use super::*;

    #[test]
    fn source_specializes_state_cardinality_and_parameters() {
        let sgd = kernel_source::<Sgd>();
        assert!(sgd.contains("float momentum;"));
        assert!(!sgd.contains("float* state_one"));
        assert!(sgd.contains(<Sgd as StatefulUpdateRule<HipC>>::BODY));

        let adam = kernel_source::<Adam>();
        assert!(adam.contains("float bias_correction_two;"));
        assert!(adam.contains("float* state_one"));
        assert!(adam.contains(<Adam as StatefulUpdateRule<HipC>>::BODY));
    }
}
