use super::prelude::prelude;

pub(crate) fn backward_source() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_backward(
    const float* output_gradient,
    const float* probabilities,
    const unsigned int* targets,
    float* logit_gradient,
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
    const float probability = probabilities[physical2(parameters.probabilities, row, column)];
    const float delta = probability - (column == (long long)target ? 1.0f : 0.0f);
    logit_gradient[physical2(parameters.logit_gradient, row, column)] +=
        (output_gradient[physical1(parameters.output_gradient, 0)] /
        (float)parameters.probabilities.shape[0]) * delta;
}}
"#,
        prelude = prelude(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_source_is_additive_and_mean_scaled() {
        let source = backward_source();
        assert!(
            source.contains("logit_gradient[physical2(parameters.logit_gradient, row, column)] +=")
        );
        assert!(source.contains("(float)parameters.probabilities.shape[0]"));
        assert!(source.contains("(long long)blockIdx.x * (long long)blockDim.x"));
    }
}
