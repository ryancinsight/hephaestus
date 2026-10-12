use super::CudaCrossEntropyScalar;
use super::prelude::prelude;

pub(crate) fn forward_source<T: CudaCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_forward(
    const {ty}* logits,
    const unsigned int* targets,
    {ty}* probabilities,
    {ty}* row_losses,
    const ForwardMeta parameters
) {{
    const long long row =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long batch = parameters.logits.shape[0];
    const long long classes = parameters.logits.shape[1];
    if (row >= batch) return;
    {ty} maximum = logits[physical2(parameters.logits, row, 0)];
    for (long long column = 1; column < classes; ++column) {{
        const {ty} value = logits[physical2(parameters.logits, row, column)];
        maximum = value > maximum ? value : maximum;
    }}
    {ty} denominator = {zero};
    for (long long column = 0; column < classes; ++column) {{
        denominator += {exp}(logits[physical2(parameters.logits, row, column)] - maximum);
    }}
    for (long long column = 0; column < classes; ++column) {{
        probabilities[physical2(parameters.probabilities, row, column)] =
            {exp}(logits[physical2(parameters.logits, row, column)] - maximum) / denominator;
    }}
    const unsigned int target = targets[physical1(parameters.targets, row)];
    row_losses[row] = {log}(denominator) +
        (maximum - logits[physical2(parameters.logits, row, (long long)target)]);
}}
"#,
        prelude = prelude::<T>(),
        ty = T::TYPE_TOKEN,
        exp = T::EXP,
        log = T::LOG,
        zero = T::ZERO,
    )
}

pub(crate) fn forward_mean_source<T: CudaCrossEntropyScalar>() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_forward_mean(
    const {ty}* row_losses,
    {ty}* loss,
    const long long batch,
    const ForwardMeta parameters
) {{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    {ty} mean = {zero};
    for (long long row = 0; row < batch; ++row) {{
        const {ty} row_loss = row_losses[row];
        mean += (row_loss - mean) / ({ty})(row + 1);
    }}
    loss[physical1(parameters.loss, 0)] = mean;
}}
"#,
        prelude = prelude::<T>(),
        ty = T::TYPE_TOKEN,
        zero = T::ZERO,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_sources_are_stable_and_layout_aware() {
        let forward = forward_source::<f32>();
        let mean = forward_mean_source::<f32>();
        assert!(forward.contains("expf(logits[physical2(parameters.logits"));
        assert!(forward.contains("row_losses[row] = logf(denominator)"));
        assert!(mean.contains("mean += (row_loss - mean) / (float)(row + 1)"));
        assert!(mean.contains("loss[physical1(parameters.loss, 0)] = mean"));
        assert!(forward.contains("(long long)blockIdx.x * (long long)blockDim.x"));
    }

    #[test]
    fn f64_forward_sources_use_native_double_arithmetic() {
        let forward = forward_source::<f64>();
        let mean = forward_mean_source::<f64>();
        assert!(forward.contains("const double* logits,"));
        assert!(forward.contains("double maximum = "));
        assert!(forward.contains("double denominator = 0.0;"));
        assert!(forward.contains("exp(logits[physical2(parameters.logits"));
        assert!(!forward.contains("expf("));
        assert!(forward.contains("row_losses[row] = log(denominator)"));
        assert!(!forward.contains("logf("));
        assert!(mean.contains("mean += (row_loss - mean) / (double)(row + 1)"));
        assert!(!forward.contains("float"));
        assert!(!mean.contains("float"));
    }
}
