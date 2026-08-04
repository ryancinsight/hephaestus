use super::prelude;
use hephaestus_core::CrossEntropyStatus;

pub(in crate::application::loss) const BACKWARD_PREFLIGHT_ENTRY: &str =
    "hephaestus_cross_entropy_backward_preflight";
pub(in crate::application::loss) const BACKWARD_ENTRY: &str = "hephaestus_cross_entropy_backward";

pub(in crate::application::loss) fn backward_source() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void {preflight_entry}(
    const float* output_gradient,
    const float* probabilities,
    const unsigned int* targets,
    const float* logit_gradient,
    unsigned int* status,
    CrossEntropyMeta parameters
) {{
    const unsigned long long global =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (global >= (unsigned long long)parameters.batch) return;
    const int row = (int)global;
    const unsigned int target = targets[physical1(parameters.targets, row)];
    if (target >= (unsigned int)parameters.classes) {{
        record_status(status, {target_status}u);
        return;
    }}
    const float upstream = output_gradient[physical1(parameters.output_gradient, 0)];
    if (!isfinite(upstream)) {{
        record_status(status, {upstream_status}u);
        return;
    }}

    float probability_sum = 0.0f;
    unsigned int row_status = 0xffffffffu;
    for (int column = 0; column < parameters.classes; ++column) {{
        const float probability =
            probabilities[physical2(parameters.probabilities, row, column)];
        if (!isfinite(probability) || probability < 0.0f || probability > 1.0f) {{
            row_status = min(row_status, {probability_status}u);
        }}
        probability_sum += probability;
        const int gradient_index = physical2(parameters.logit_gradient, row, column);
        const float current = logit_gradient[gradient_index];
        if (!isfinite(current)) {{
            row_status = min(row_status, {destination_status}u);
        }}
        const float indicator = column == (int)target ? 1.0f : 0.0f;
        const float candidate = current
            + upstream * (probability - indicator) / (float)parameters.batch;
        if (!isfinite(candidate)) {{
            row_status = min(row_status, {arithmetic_status}u);
        }}
    }}
    if (!isfinite(probability_sum)
        || fabsf(probability_sum - 1.0f) > parameters.probability_tolerance) {{
        row_status = min(row_status, {probability_status}u);
    }}
    if (row_status != 0xffffffffu) {{
        record_status(status, row_status);
    }}
}}

extern "C" __global__ void {backward_entry}(
    const float* output_gradient,
    const float* probabilities,
    const unsigned int* targets,
    float* logit_gradient,
    CrossEntropyMeta parameters
) {{
    const unsigned long long global =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements =
        (unsigned long long)parameters.batch * (unsigned long long)parameters.classes;
    if (global >= elements) return;
    const int linear = (int)global;
    const int row = linear / parameters.classes;
    const int column = linear % parameters.classes;
    const unsigned int target = targets[physical1(parameters.targets, row)];
    const float upstream = output_gradient[physical1(parameters.output_gradient, 0)];
    const float probability = probabilities[physical2(parameters.probabilities, row, column)];
    const float indicator = column == (int)target ? 1.0f : 0.0f;
    const int destination = physical2(parameters.logit_gradient, row, column);
    logit_gradient[destination] +=
        upstream * (probability - indicator) / (float)parameters.batch;
}}
"#,
        prelude = prelude::source(),
        preflight_entry = BACKWARD_PREFLIGHT_ENTRY,
        backward_entry = BACKWARD_ENTRY,
        target_status = CrossEntropyStatus::TargetOutOfRange.code(),
        upstream_status = CrossEntropyStatus::NonFiniteOutputGradient.code(),
        probability_status = CrossEntropyStatus::InvalidProbabilities.code(),
        destination_status = CrossEntropyStatus::NonFiniteGradientDestination.code(),
        arithmetic_status = CrossEntropyStatus::NonFiniteBackwardArithmetic.code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_preflights_additive_candidates_before_parallel_mutation() {
        let source = backward_source();
        assert!(source.contains("const float candidate = current"));
        assert!(source.contains("fabsf(probability_sum - 1.0f)"));
        assert!(source.contains("logit_gradient[destination] +="));
        assert!(source.contains("parameters.probability_tolerance"));
        assert!(source.contains("row_status = min(row_status"));
        assert!(source.contains("(unsigned long long)blockIdx.x * blockDim.x"));
        assert!(source.contains("if (global >= elements) return;"));
    }
}
