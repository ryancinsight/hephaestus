use super::HipCrossEntropyScalar;
use super::prelude;
use hephaestus_core::CrossEntropyStatus;

pub(in crate::application::loss) const BACKWARD_PREFLIGHT_ENTRY: &str =
    "hephaestus_cross_entropy_backward_preflight";
pub(in crate::application::loss) const BACKWARD_ENTRY: &str = "hephaestus_cross_entropy_backward";

pub(in crate::application::loss) fn backward_source<T: HipCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void {preflight_entry}(
    const {ty}* output_gradient,
    const {ty}* probabilities,
    const unsigned int* targets,
    const {ty}* logit_gradient,
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
    const {ty} upstream = output_gradient[physical1(parameters.output_gradient, 0)];
    if (!isfinite(upstream)) {{
        record_status(status, {upstream_status}u);
        return;
    }}

    {ty} probability_sum = {zero};
    unsigned int row_status = 0xffffffffu;
    for (int column = 0; column < parameters.classes; ++column) {{
        const {ty} probability =
            probabilities[physical2(parameters.probabilities, row, column)];
        if (!isfinite(probability) || probability < {zero} || probability > {one}) {{
            row_status = min(row_status, {probability_status}u);
        }}
        probability_sum += probability;
        const int gradient_index = physical2(parameters.logit_gradient, row, column);
        const {ty} current = logit_gradient[gradient_index];
        if (!isfinite(current)) {{
            row_status = min(row_status, {destination_status}u);
        }}
        const {ty} indicator = column == (int)target ? {one} : {zero};
        const {ty} candidate = current
            + upstream * (probability - indicator) / ({ty})parameters.batch;
        if (!isfinite(candidate)) {{
            row_status = min(row_status, {arithmetic_status}u);
        }}
    }}
    if (!isfinite(probability_sum)
        || {fabs}(probability_sum - {one}) > parameters.probability_tolerance) {{
        row_status = min(row_status, {probability_status}u);
    }}
    if (row_status != 0xffffffffu) {{
        record_status(status, row_status);
    }}
}}

extern "C" __global__ void {backward_entry}(
    const {ty}* output_gradient,
    const {ty}* probabilities,
    const unsigned int* targets,
    {ty}* logit_gradient,
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
    const {ty} upstream = output_gradient[physical1(parameters.output_gradient, 0)];
    const {ty} probability = probabilities[physical2(parameters.probabilities, row, column)];
    const {ty} indicator = column == (int)target ? {one} : {zero};
    const int destination = physical2(parameters.logit_gradient, row, column);
    logit_gradient[destination] +=
        upstream * (probability - indicator) / ({ty})parameters.batch;
}}
"#,
        prelude = prelude::source::<T>(),
        preflight_entry = BACKWARD_PREFLIGHT_ENTRY,
        backward_entry = BACKWARD_ENTRY,
        ty = T::TYPE_TOKEN,
        fabs = T::FABS,
        one = T::ONE,
        zero = T::ZERO,
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
        let source = backward_source::<f32>();
        assert!(source.contains("const float candidate = current"));
        assert!(source.contains("fabsf(probability_sum - 1.0f)"));
        assert!(source.contains("logit_gradient[destination] +="));
        assert!(source.contains("parameters.probability_tolerance"));
        assert!(source.contains("row_status = min(row_status"));
        assert!(source.contains("(unsigned long long)blockIdx.x * blockDim.x"));
        assert!(source.contains("if (global >= elements) return;"));
    }

    #[test]
    fn f64_source_uses_native_double_arithmetic() {
        let source = backward_source::<f64>();
        assert!(source.contains("const double* output_gradient,"));
        assert!(source.contains("const double upstream = "));
        assert!(source.contains("double probability_sum = 0.0;"));
        assert!(source.contains("fabs(probability_sum - 1.0)"));
        assert!(!source.contains("fabsf("));
        assert!(source.contains("const double indicator = column == (int)target ? 1.0 : 0.0;"));
        assert!(source.contains("upstream * (probability - indicator) / (double)parameters.batch"));
        assert!(!source.contains("float*"));
        assert!(!source.contains("float "));
        assert!(!source.contains("(float)"));
        assert!(!source.contains("1.0f"));
    }
}
