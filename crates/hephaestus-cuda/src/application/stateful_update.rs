//! Monomorphized CUDA stateful parameter updates.

use hephaestus_core::{
    BlockWidth, CudaC, DialectScalar, Result, StatefulUpdateAliasing, StatefulUpdateMeta,
    StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, plan_stateful_update,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::{CudaBuffer, CudaDevice};

const ENTRY: &str = "stateful_update_kernel";

fn parameters_declaration<Rule, T>() -> String
where
    Rule: StatefulUpdateRule<CudaC>,
    T: DialectScalar<CudaC>,
{
    let ty = T::TYPE_TOKEN;
    let fields = Rule::parameter_fields_for::<T>()
        .iter()
        .map(|field| format!("    {ty} {field};\n"))
        .collect::<String>();
    format!("struct Parameters {{\n{fields}}};\n")
}

fn kernel_source<Rule, T>() -> String
where
    Rule: StatefulUpdateRule<CudaC>,
    T: DialectScalar<CudaC>,
{
    let ty = T::TYPE_TOKEN;
    let state_one_parameter = if Rule::STATE_COUNT == 2 {
        format!(",\n    {ty}* state_one")
    } else {
        String::new()
    };
    let state_one_offset = if Rule::STATE_COUNT == 2 {
        "    int state_one_offset = (int)lmeta.offsets[3];\n"
    } else {
        ""
    };
    let state_one_stride = if Rule::STATE_COUNT == 2 {
        "        state_one_offset += index * lmeta.strides[3][lane][component];\n"
    } else {
        ""
    };
    let state_one_load = if Rule::STATE_COUNT == 2 {
        format!(
            "    const {ty} state_one_value = state_one[state_one_offset];\n    {ty} state_one_next = state_one_value;\n"
        )
    } else {
        String::new()
    };
    let state_one_store = if Rule::STATE_COUNT == 2 {
        "    state_one[state_one_offset] = state_one_next;\n"
    } else {
        ""
    };
    format!(
        r#"struct Meta {{
    unsigned int shape[2][4];
    int strides[4][2][4];
    unsigned int offsets[4];
    unsigned int dispatch[4];
}};
{parameters}
extern "C" __global__ void {entry}(
    Meta lmeta,
    Parameters parameters,
    {ty}* parameter,
    const {ty}* gradient,
    {ty}* state_zero{state_one_parameter}
) {{
    const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= lmeta.dispatch[0]) {{
        return;
    }}
    unsigned int remainder = linear;
    int parameter_offset = (int)lmeta.offsets[0];
    int gradient_offset = (int)lmeta.offsets[1];
    int state_zero_offset = (int)lmeta.offsets[2];
{state_one_offset}    for (int dimension = 7; dimension >= 0; --dimension) {{
        const int lane = dimension / 4;
        const int component = dimension % 4;
        const unsigned int extent = lmeta.shape[lane][component];
        const int index = (int)(remainder % extent);
        remainder /= extent;
        parameter_offset += index * lmeta.strides[0][lane][component];
        gradient_offset += index * lmeta.strides[1][lane][component];
        state_zero_offset += index * lmeta.strides[2][lane][component];
{state_one_stride}    }}
    const {ty} parameter_value = parameter[parameter_offset];
    const {ty} gradient_value = gradient[gradient_offset];
    const {ty} state_zero_value = state_zero[state_zero_offset];
    {ty} parameter_next = parameter_value;
    {ty} state_zero_next = state_zero_value;
{state_one_load}    {body}
    parameter[parameter_offset] = parameter_next;
    state_zero[state_zero_offset] = state_zero_next;
{state_one_store}}}
"#,
        parameters = parameters_declaration::<Rule, T>(),
        entry = ENTRY,
        ty = T::TYPE_TOKEN,
        body = Rule::body_for::<T>(),
    )
}

fn aliasing<T, const N: usize>(
    operands: &StatefulUpdateOperands<'_, CudaBuffer<T>, N>,
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

fn launch_stateful_update<Rule, T, P, const N: usize>(
    device: &CudaDevice,
    operands: StatefulUpdateOperands<'_, CudaBuffer<T>, N>,
    parameters: P,
    meta: StatefulUpdateMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Rule: StatefulUpdateRule<CudaC>,
    T: DialectScalar<CudaC>,
{
    let key = PipelineKey::StatefulUpdate {
        rule: core::any::TypeId::of::<Rule>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, ENTRY, kernel_source::<Rule, T>)?;
    let state_zero = operands
        .states
        .first()
        .expect("invariant: planner validated state zero");
    let mut meta = meta;
    let mut parameters = parameters;
    let mut parameter_ptr = operands.parameter.buffer.raw();
    let mut gradient_ptr = operands.gradient.buffer.raw();
    let mut state_zero_ptr = state_zero.buffer.raw();
    let mut common: [*mut core::ffi::c_void; 5] = [
        (&mut meta as *mut StatefulUpdateMeta).cast(),
        (&mut parameters as *mut P).cast(),
        (&mut parameter_ptr as *mut u64).cast(),
        (&mut gradient_ptr as *mut u64).cast(),
        (&mut state_zero_ptr as *mut u64).cast(),
    ];
    let config = LaunchConfig::linear(grid_size(len, width)?, width);
    if let Some(state_one) = operands.states.get(1) {
        let mut state_one_ptr = state_one.buffer.raw();
        let mut args = [
            common[0],
            common[1],
            common[2],
            common[3],
            common[4],
            (&mut state_one_ptr as *mut u64).cast(),
        ];
        launch_kernel(device, &kernel, config, &mut args)
    } else {
        launch_kernel(device, &kernel, config, &mut common)
    }
}

/// Provider-owned CUDA implementation of stateful parameter updates.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaStatefulUpdateOps;

fn validate_device<T, const N: usize>(
    device: &CudaDevice,
    operands: &StatefulUpdateOperands<'_, CudaBuffer<T>, N>,
) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    #[cfg(feature = "cuda")]
    let matches = |buffer: &CudaBuffer<T>| {
        buffer
            .context
            .as_ref()
            .is_some_and(|context| std::sync::Arc::ptr_eq(context, device.cuda_context()))
    };
    #[cfg(not(feature = "cuda"))]
    let matches = |_buffer: &CudaBuffer<T>| true;

    if matches(operands.parameter.buffer)
        && matches(operands.gradient.buffer)
        && operands.states.iter().all(|state| matches(state.buffer))
    {
        Ok(())
    } else {
        Err(hephaestus_core::HephaestusError::InvalidConfiguration {
            message: "CUDA stateful-update operands must belong to the dispatch device".to_string(),
        })
    }
}

/// Shared validate-plan-launch driver: both widths run the same pipeline
/// with width-selected kernel text, parameter bytes, and cache key.
fn dispatch<Rule, T, P, const N: usize>(
    device: &CudaDevice,
    operands: StatefulUpdateOperands<'_, CudaBuffer<T>, N>,
    parameters: P,
    validate: fn(&P) -> Result<()>,
) -> Result<()>
where
    Rule: StatefulUpdateRule<CudaC>,
    T: DialectScalar<CudaC>,
{
    validate(&parameters)?;
    validate_device(device, &operands)?;
    let plan = plan_stateful_update(operands, Rule::STATE_COUNT, aliasing(&operands))?;
    if plan.is_empty() {
        return Ok(());
    }
    launch_stateful_update::<Rule, T, P, N>(
        device,
        operands,
        parameters,
        plan.metadata(),
        BlockWidth::DEFAULT,
        plan.len(),
    )
}

impl StatefulUpdateOps<CudaDevice> for CudaStatefulUpdateOps {
    type Dialect = CudaC;

    fn stateful_update<Rule, const N: usize>(
        &self,
        device: &CudaDevice,
        operands: StatefulUpdateOperands<'_, CudaBuffer<f32>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        dispatch::<Rule, f32, _, N>(device, operands, parameters, Rule::validate_parameters)
    }

    fn stateful_update_f64<Rule, const N: usize>(
        &self,
        device: &CudaDevice,
        operands: StatefulUpdateOperands<'_, CudaBuffer<f64>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::ParametersF64,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        dispatch::<Rule, f64, _, N>(device, operands, parameters, Rule::validate_parameters_f64)
    }
}

#[cfg(test)]
mod tests {
    use hephaestus_core::{Adam, Sgd};

    use super::*;

    #[test]
    fn source_specializes_state_cardinality_and_parameters() {
        let sgd = kernel_source::<Sgd, f32>();
        assert!(sgd.contains("float momentum;"));
        assert!(!sgd.contains("float* state_one"));

        let adam = kernel_source::<Adam, f32>();
        assert!(adam.contains("float bias_correction_two;"));
        assert!(adam.contains("float* state_one"));
        assert!(adam.contains(<Adam as StatefulUpdateRule<CudaC>>::BODY));
    }

    #[test]
    fn f64_source_types_storage_and_parameters_double() {
        let sgd = kernel_source::<Sgd, f64>();
        assert!(sgd.contains("double momentum;"));
        assert!(sgd.contains("double* parameter,"));
        assert!(sgd.contains("const double* gradient,"));
        assert!(!sgd.contains("float"));

        let adam = kernel_source::<Adam, f64>();
        assert!(adam.contains("double bias_correction_two;"));
        assert!(adam.contains("double* state_one"));
    }

    #[test]
    fn f64_adam_selects_the_double_literal_body() {
        let adam = kernel_source::<Adam, f64>();
        let body = <Adam as StatefulUpdateRule<CudaC>>::body_for::<f64>();
        assert!(adam.contains(body));
        assert!(!body.contains("1.0f"));
    }
}
