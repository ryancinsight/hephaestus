use super::CudaCrossEntropyScalar;
use super::prelude::prelude;

pub(crate) fn backward_source<T: CudaCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_backward(
    const {ty}* output_gradient,
    const {ty}* probabilities,
    const unsigned int* targets,
    {ty}* logit_gradient,
    const BackwardMeta parameters
) {{
    const long long linear =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long classes = parameters.probabilities.shape[1];
    const long long elements = parameters.probabilities.shape[0] * classes;
    if (linear >= elements) return;
    const long long row = linear / classes;
    const long long column = linear % classes;
    const unsigned int target = targets[physical1(parameters.targets, row)];
    const {ty} probability = probabilities[physical2(parameters.probabilities, row, column)];
    const {ty} delta = probability - (column == (long long)target ? {one} : {zero});
    logit_gradient[physical2(parameters.logit_gradient, row, column)] +=
        (output_gradient[physical1(parameters.output_gradient, 0)] /
        ({ty})parameters.probabilities.shape[0]) * delta;
}}
"#,
        prelude = prelude::<T>(),
        ty = T::TYPE_TOKEN,
        one = T::ONE,
        zero = T::ZERO,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_source_is_additive_and_mean_scaled() {
        let source = backward_source::<f32>();
        assert!(
            source.contains("logit_gradient[physical2(parameters.logit_gradient, row, column)] +=")
        );
        assert!(source.contains("(float)parameters.probabilities.shape[0]"));
        assert!(source.contains("(long long)blockIdx.x * (long long)blockDim.x"));
    }

    #[test]
    fn f64_backward_source_uses_native_double_arithmetic() {
        let source = backward_source::<f64>();
        assert!(source.contains("const double* probabilities,"));
        assert!(source.contains(
            "const double delta = probability - (column == (long long)target ? 1.0 : 0.0);"
        ));
        assert!(source.contains("(double)parameters.probabilities.shape[0]"));
        assert!(!source.contains("float"));
        assert!(!source.contains("1.0f"));
    }
}
