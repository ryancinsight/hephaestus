use super::prelude::prelude;
use hephaestus_core::CrossEntropyStatus;

pub(crate) fn forward_preflight_source() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_forward_preflight(
    const float* logits,
    const unsigned int* targets,
    unsigned int* status,
    const ForwardMeta parameters
) {{
    const long long row =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long batch = parameters.logits.shape[0];
    const long long classes = parameters.logits.shape[1];
    if (row >= batch) return;
    const unsigned int target = targets[physical1(parameters.targets, row)];
    if ((unsigned long long)target >= (unsigned long long)classes) {{
        cross_entropy_fail(status, {invalid_target}u);
        return;
    }}
    float maximum = logits[physical2(parameters.logits, row, 0)];
    if (!isfinite(maximum)) {{
        cross_entropy_fail(status, {nonfinite_logits}u);
        return;
    }}
    for (long long column = 1; column < classes; ++column) {{
        const float value = logits[physical2(parameters.logits, row, column)];
        if (!isfinite(value)) {{
            cross_entropy_fail(status, {nonfinite_logits}u);
            return;
        }}
        maximum = value > maximum ? value : maximum;
    }}
    float denominator = 0.0f;
    for (long long column = 0; column < classes; ++column) {{
        denominator += expf(logits[physical2(parameters.logits, row, column)] - maximum);
    }}
    const float target_logit = logits[physical2(parameters.logits, row, (long long)target)];
    const float row_loss = logf(denominator) + (maximum - target_logit);
    if (!(denominator > 0.0f) || !isfinite(denominator) || !isfinite(row_loss)) {{
        cross_entropy_fail(status, {arithmetic}u);
    }}
}}
"#,
        prelude = prelude(),
        invalid_target = CrossEntropyStatus::TargetOutOfRange.code(),
        nonfinite_logits = CrossEntropyStatus::NonFiniteLogits.code(),
        arithmetic = CrossEntropyStatus::NonFiniteForwardArithmetic.code(),
    )
}

pub(crate) fn backward_preflight_source() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_backward_preflight(
    const float* output_gradient,
    const float* probabilities,
    const unsigned int* targets,
    const float* logit_gradient,
    unsigned int* status,
    const BackwardMeta parameters
) {{
    const long long row =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long batch = parameters.probabilities.shape[0];
    const long long classes = parameters.probabilities.shape[1];
    if (row >= batch) return;
    const unsigned int target = targets[physical1(parameters.targets, row)];
    if ((unsigned long long)target >= (unsigned long long)classes) {{
        cross_entropy_fail(status, {invalid_target}u);
        return;
    }}
    const float upstream = output_gradient[physical1(parameters.output_gradient, 0)];
    if (!isfinite(upstream)) {{
        cross_entropy_fail(status, {nonfinite_upstream}u);
        return;
    }}
    float sum = 0.0f;
    unsigned int row_status = 0xffffffffu;
    for (long long column = 0; column < classes; ++column) {{
        const float probability = probabilities[physical2(parameters.probabilities, row, column)];
        if (!isfinite(probability) || probability < 0.0f || probability > 1.0f) {{
            row_status = min(row_status, {invalid_probabilities}u);
        }}
        sum += probability;
        const float delta = probability - (column == (long long)target ? 1.0f : 0.0f);
        const float increment = (upstream / (float)batch) * delta;
        const float current = logit_gradient[physical2(parameters.logit_gradient, row, column)];
        if (!isfinite(current)) {{
            row_status = min(row_status, {nonfinite_gradient}u);
        }}
        if (!isfinite(increment) || !isfinite(current + increment)) {{
            row_status = min(row_status, {arithmetic}u);
        }}
    }}
    if (!isfinite(sum) || fabsf(sum - 1.0f) > parameters.tolerance) {{
        row_status = min(row_status, {invalid_probabilities}u);
    }}
    if (row_status != 0xffffffffu) {{
        cross_entropy_fail(status, row_status);
    }}
}}
"#,
        prelude = prelude(),
        nonfinite_upstream = CrossEntropyStatus::NonFiniteOutputGradient.code(),
        invalid_target = CrossEntropyStatus::TargetOutOfRange.code(),
        invalid_probabilities = CrossEntropyStatus::InvalidProbabilities.code(),
        nonfinite_gradient = CrossEntropyStatus::NonFiniteGradientDestination.code(),
        arithmetic = CrossEntropyStatus::NonFiniteBackwardArithmetic.code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_sources_validate_before_writes() {
        let forward = forward_preflight_source();
        let backward = backward_preflight_source();
        assert!(forward.contains("logf(denominator) + (maximum - target_logit)"));
        assert!(forward.contains("atomicMin(status, code)"));
        assert!(backward.contains("fabsf(sum - 1.0f) > parameters.tolerance"));
        let target = backward
            .find("const unsigned int target")
            .expect("target check");
        let upstream = backward
            .find("const float upstream")
            .expect("upstream check");
        assert!(target < upstream, "target status has canonical priority");
        assert!(backward.contains("row_status = min(row_status"));
        assert!(!forward.contains("probabilities["));
        assert!(
            !backward
                .contains("logit_gradient[physical2(parameters.logit_gradient, row, column)] +=")
        );
    }
}
