use super::prelude::prelude;

pub(crate) fn forward_source() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_forward(
    const float* logits,
    const unsigned int* targets,
    float* probabilities,
    float* row_losses,
    const ForwardMeta parameters
) {{
    const long long row =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long batch = parameters.logits.shape[0];
    const long long classes = parameters.logits.shape[1];
    if (row >= batch) return;
    float maximum = logits[physical2(parameters.logits, row, 0)];
    for (long long column = 1; column < classes; ++column) {{
        const float value = logits[physical2(parameters.logits, row, column)];
        maximum = value > maximum ? value : maximum;
    }}
    float denominator = 0.0f;
    for (long long column = 0; column < classes; ++column) {{
        denominator += expf(logits[physical2(parameters.logits, row, column)] - maximum);
    }}
    for (long long column = 0; column < classes; ++column) {{
        probabilities[physical2(parameters.probabilities, row, column)] =
            expf(logits[physical2(parameters.logits, row, column)] - maximum) / denominator;
    }}
    const unsigned int target = targets[physical1(parameters.targets, row)];
    row_losses[row] = logf(denominator) +
        (maximum - logits[physical2(parameters.logits, row, (long long)target)]);
}}
"#,
        prelude = prelude(),
    )
}

pub(crate) fn forward_mean_source() -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void cross_entropy_forward_mean(
    const float* row_losses,
    float* loss,
    const long long batch,
    const ForwardMeta parameters
) {{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float mean = 0.0f;
    for (long long row = 0; row < batch; ++row) {{
        const float row_loss = row_losses[row];
        mean += (row_loss - mean) / (float)(row + 1);
    }}
    loss[physical1(parameters.loss, 0)] = mean;
}}
"#,
        prelude = prelude(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_sources_are_stable_and_layout_aware() {
        let forward = forward_source();
        let mean = forward_mean_source();
        assert!(forward.contains("expf(logits[physical2(parameters.logits"));
        assert!(forward.contains("row_losses[row] = logf(denominator)"));
        assert!(mean.contains("mean += (row_loss - mean) / (float)(row + 1)"));
        assert!(mean.contains("loss[physical1(parameters.loss, 0)] = mean"));
        assert!(forward.contains("(long long)blockIdx.x * (long long)blockDim.x"));
    }
}
