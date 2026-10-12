use super::CudaCrossEntropyScalar;
use super::prelude::prelude;
use hephaestus_core::CrossEntropyStatus;

pub(crate) fn forward_preflight_source<T: CudaCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_forward_preflight(
    const {ty}* logits,
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
    {ty} maximum = logits[physical2(parameters.logits, row, 0)];
    if (!isfinite(maximum)) {{
        cross_entropy_fail(status, {nonfinite_logits}u);
        return;
    }}
    for (long long column = 1; column < classes; ++column) {{
        const {ty} value = logits[physical2(parameters.logits, row, column)];
        if (!isfinite(value)) {{
            cross_entropy_fail(status, {nonfinite_logits}u);
            return;
        }}
        maximum = value > maximum ? value : maximum;
    }}
    {ty} denominator = {zero};
    for (long long column = 0; column < classes; ++column) {{
        denominator += {exp}(logits[physical2(parameters.logits, row, column)] - maximum);
    }}
    const {ty} target_logit = logits[physical2(parameters.logits, row, (long long)target)];
    const {ty} row_loss = {log}(denominator) + (maximum - target_logit);
    if (!(denominator > {zero}) || !isfinite(denominator) || !isfinite(row_loss)) {{
        cross_entropy_fail(status, {arithmetic}u);
    }}
}}
"#,
        prelude = prelude::<T>(),
        ty = T::TYPE_TOKEN,
        exp = T::EXP,
        log = T::LOG,
        zero = T::ZERO,
        invalid_target = CrossEntropyStatus::TargetOutOfRange.code(),
        nonfinite_logits = CrossEntropyStatus::NonFiniteLogits.code(),
        arithmetic = CrossEntropyStatus::NonFiniteForwardArithmetic.code(),
    )
}

pub(crate) fn backward_preflight_source<T: CudaCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_backward_preflight(
    const {ty}* output_gradient,
    const {ty}* probabilities,
    const unsigned int* targets,
    const {ty}* logit_gradient,
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
    const {ty} upstream = output_gradient[physical1(parameters.output_gradient, 0)];
    if (!isfinite(upstream)) {{
        cross_entropy_fail(status, {nonfinite_upstream}u);
        return;
    }}
    {ty} sum = {zero};
    unsigned int row_status = 0xffffffffu;
    for (long long column = 0; column < classes; ++column) {{
        const {ty} probability = probabilities[physical2(parameters.probabilities, row, column)];
        if (!isfinite(probability) || probability < {zero} || probability > {one}) {{
            row_status = min(row_status, {invalid_probabilities}u);
        }}
        sum += probability;
        const {ty} delta = probability - (column == (long long)target ? {one} : {zero});
        const {ty} increment = (upstream / ({ty})batch) * delta;
        const {ty} current = logit_gradient[physical2(parameters.logit_gradient, row, column)];
        if (!isfinite(current)) {{
            row_status = min(row_status, {nonfinite_gradient}u);
        }}
        if (!isfinite(increment) || !isfinite(current + increment)) {{
            row_status = min(row_status, {arithmetic}u);
        }}
    }}
    if (!isfinite(sum) || {fabs}(sum - {one}) > parameters.tolerance) {{
        row_status = min(row_status, {invalid_probabilities}u);
    }}
    if (row_status != 0xffffffffu) {{
        cross_entropy_fail(status, row_status);
    }}
}}
"#,
        prelude = prelude::<T>(),
        ty = T::TYPE_TOKEN,
        fabs = T::FABS,
        one = T::ONE,
        zero = T::ZERO,
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
        let forward = forward_preflight_source::<f32>();
        let backward = backward_preflight_source::<f32>();
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

    #[test]
    fn f64_preflight_sources_use_native_double_arithmetic() {
        let forward = forward_preflight_source::<f64>();
        let backward = backward_preflight_source::<f64>();
        assert!(forward.contains("const double* logits,"));
        assert!(forward.contains("log(denominator) + (maximum - target_logit)"));
        assert!(!forward.contains("logf("));
        assert!(!forward.contains("expf("));
        assert!(backward.contains("fabs(sum - 1.0) > parameters.tolerance"));
        assert!(!backward.contains("fabsf("));
        assert!(backward.contains("const double upstream"));
        assert!(backward.contains("double tolerance;"));
        assert!(!forward.contains("float"));
        assert!(!backward.contains("float"));
    }
}
