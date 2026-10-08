use super::HipCrossEntropyScalar;
use super::prelude;
use hephaestus_core::CrossEntropyStatus;

pub(in crate::application::loss) const FORWARD_PREFLIGHT_ENTRY: &str =
    "hephaestus_cross_entropy_forward_preflight";
pub(in crate::application::loss) const FORWARD_ENTRY: &str = "hephaestus_cross_entropy_forward";
pub(in crate::application::loss) const FORWARD_MEAN_ENTRY: &str =
    "hephaestus_cross_entropy_forward_mean";

pub(in crate::application::loss) fn forward_source<T: HipCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __device__ __forceinline__ bool cross_entropy_row(
    const {ty}* logits,
    const unsigned int* targets,
    int row,
    CrossEntropyMeta parameters,
    {ty}* maximum_out,
    {ty}* denominator_out,
    {ty}* loss_out
) {{
    const unsigned int target = targets[physical1(parameters.targets, row)];
    if (target >= (unsigned int)parameters.classes) return false;

    {ty} maximum = -INFINITY;
    for (int column = 0; column < parameters.classes; ++column) {{
        const {ty} value = logits[physical2(parameters.logits, row, column)];
        if (!isfinite(value)) return false;
        maximum = {fmax}(maximum, value);
    }}
    {ty} denominator = {zero};
    for (int column = 0; column < parameters.classes; ++column) {{
        denominator += {exp}(logits[physical2(parameters.logits, row, column)] - maximum);
    }}
    const {ty} target_logit = logits[physical2(parameters.logits, row, (int)target)];
    const {ty} loss = {log}(denominator) + (maximum - target_logit);
    if (!isfinite(denominator) || denominator <= {zero} || !isfinite(loss)) return false;
    *maximum_out = maximum;
    *denominator_out = denominator;
    *loss_out = loss;
    return true;
}}

extern "C" __global__ void {preflight_entry}(
    const {ty}* logits,
    const unsigned int* targets,
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
    {ty} maximum;
    {ty} denominator;
    {ty} loss;
    if (!cross_entropy_row(
        logits, targets, row, parameters, &maximum, &denominator, &loss
    )) {{
        bool finite = true;
        for (int column = 0; column < parameters.classes; ++column) {{
            finite = finite && isfinite(logits[physical2(parameters.logits, row, column)]);
        }}
        record_status(
            status,
            finite ? {arithmetic_status}u : {logits_status}u
        );
    }}
}}

extern "C" __global__ void {forward_entry}(
    const {ty}* logits,
    const unsigned int* targets,
    {ty}* probabilities,
    {ty}* row_losses,
    CrossEntropyMeta parameters
) {{
    const unsigned long long global =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (global >= (unsigned long long)parameters.batch) return;
    const int row = (int)global;
    {ty} maximum;
    {ty} denominator;
    {ty} loss;
    if (!cross_entropy_row(
        logits, targets, row, parameters, &maximum, &denominator, &loss
    )) return;
    for (int column = 0; column < parameters.classes; ++column) {{
        probabilities[physical2(parameters.probabilities, row, column)] =
            {exp}(logits[physical2(parameters.logits, row, column)] - maximum) / denominator;
    }}
    row_losses[row] = loss;
}}

extern "C" __global__ void {mean_entry}(
    const {ty}* row_losses,
    {ty}* loss,
    CrossEntropyMeta parameters
) {{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    {ty} mean = {zero};
    for (int row = 0; row < parameters.batch; ++row) {{
        mean += (row_losses[row] - mean) / ({ty})(row + 1);
    }}
    loss[physical1(parameters.loss, 0)] = mean;
}}
"#,
        prelude = prelude::source::<T>(),
        preflight_entry = FORWARD_PREFLIGHT_ENTRY,
        forward_entry = FORWARD_ENTRY,
        mean_entry = FORWARD_MEAN_ENTRY,
        ty = T::TYPE_TOKEN,
        exp = T::EXP,
        log = T::LOG,
        fmax = T::FMAX,
        zero = T::ZERO,
        target_status = CrossEntropyStatus::TargetOutOfRange.code(),
        arithmetic_status = CrossEntropyStatus::NonFiniteForwardArithmetic.code(),
        logits_status = CrossEntropyStatus::NonFiniteLogits.code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_uses_stable_softmax_and_online_mean() {
        let source = forward_source::<f32>();
        assert!(
            source.contains("expf(logits[physical2(parameters.logits, row, column)] - maximum)")
        );
        assert!(source.contains("logf(denominator) + (maximum - target_logit)"));
        assert!(source.contains("mean += (row_losses[row] - mean) / (float)(row + 1)"));
        assert!(source.contains("(unsigned long long)blockIdx.x * blockDim.x"));
        assert!(source.contains("if (global >= (unsigned long long)parameters.batch) return;"));
        assert!(!source.contains("malloc"));
    }

    #[test]
    fn preflight_precedes_separate_mutation_entries() {
        let source = forward_source::<f32>();
        let preflight_signature = format!("void {FORWARD_PREFLIGHT_ENTRY}(");
        let forward_signature = format!("void {FORWARD_ENTRY}(");
        let preflight = source.find(&preflight_signature).expect("preflight entry");
        let mutation = source.find(&forward_signature).expect("forward entry");
        assert!(preflight < mutation);
        let preflight_body = &source[preflight..mutation];
        assert!(!preflight_body.contains("probabilities["));
        assert!(!preflight_body.contains("row_losses["));
    }

    #[test]
    fn f64_source_uses_native_double_arithmetic() {
        let source = forward_source::<f64>();
        assert!(source.contains("const double* logits,"));
        assert!(source.contains("double maximum = -INFINITY;"));
        assert!(source.contains("maximum = fmax(maximum, value);"));
        assert!(source.contains("double denominator = 0.0;"));
        assert!(
            source.contains("exp(logits[physical2(parameters.logits, row, column)] - maximum)")
        );
        assert!(!source.contains("expf("));
        assert!(source.contains("log(denominator) + (maximum - target_logit)"));
        assert!(!source.contains("logf("));
        assert!(source.contains("mean += (row_losses[row] - mean) / (double)(row + 1)"));
        assert!(source.contains("double probability_tolerance;"));
        assert!(!source.contains("float*"));
        assert!(!source.contains("float "));
        assert!(!source.contains("(float)"));
        assert!(!source.contains("0.0f"));
    }
}
