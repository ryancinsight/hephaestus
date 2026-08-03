use hephaestus_core::AttentionSemanticStatus;

use super::prelude::prelude;

pub(crate) fn forward_preflight_source(scalar: &str, exponential: &str) -> String {
    format!(
        r#"{prelude}
extern "C" __device__ __forceinline__ bool attention_finite3(
    const {scalar}* values,
    const LayoutMeta layout,
    const long long linear
) {{
    const long long count = layout.shape[0] * layout.shape[1] * layout.shape[2];
    if (linear >= count) return true;
    const long long third = linear % layout.shape[2];
    const long long row = linear / layout.shape[2];
    const long long second = row % layout.shape[1];
    const long long first = row / layout.shape[1];
    return isfinite(values[physical3(layout, first, second, third)]);
}}

extern "C" __device__ __forceinline__ bool attention_finite2(
    const {scalar}* values,
    const LayoutMeta layout,
    const long long linear
) {{
    const long long count = layout.shape[0] * layout.shape[1];
    if (linear >= count) return true;
    const long long second = linear % layout.shape[1];
    const long long first = linear / layout.shape[1];
    return isfinite(values[physical2(layout, first, second)]);
}}

extern "C" __device__ __forceinline__ bool attention_active(
    const {scalar}* keep,
    const ForwardMeta parameters,
    const long long batch,
    const long long query_index,
    const long long key_index
) {{
    const bool causal_keep = parameters.causal == 0 || key_index <= query_index;
    const bool grouped_keep = parameters.keep_present == 0 ||
        keep[physical2(parameters.keep, batch / parameters.heads_per_batch, key_index)] != ({scalar})0;
    return causal_keep && grouped_keep;
}}

extern "C" __device__ __forceinline__ {scalar} attention_score(
    const {scalar}* query,
    const {scalar}* key,
    const ForwardMeta parameters,
    const long long batch,
    const long long query_index,
    const long long key_index,
    const {scalar} scale
) {{
    {scalar} score = ({scalar})0;
    for (long long feature = 0; feature < parameters.query.shape[2]; ++feature) {{
        score += query[physical3(parameters.query, batch, query_index, feature)] *
            key[physical3(parameters.key, batch, key_index, feature)];
    }}
    return score * scale;
}}

extern "C" __device__ __forceinline__ {scalar} attention_convex(
    const {scalar} current,
    const {scalar} value,
    const {scalar} value_weight
) {{
    if ((current >= ({scalar})0) == (value >= ({scalar})0)) {{
        if (current <= value) {{
            return current + value_weight * (value - current);
        }}
        return value + (({scalar})1 - value_weight) * (current - value);
    }}
    return (({scalar})1 - value_weight) * current + value_weight * value;
}}

extern "C" __global__ void attention_forward_preflight(
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* keep,
    unsigned int* status,
    const {scalar} scale,
    const ForwardMeta parameters
) {{
    const long long linear = (long long)(blockIdx.x * blockDim.x + threadIdx.x);
    if (!attention_finite3(query, parameters.query, linear))
        attention_fail(status, {nonfinite_query}u);
    if (!attention_finite3(key, parameters.key, linear))
        attention_fail(status, {nonfinite_key}u);
    if (!attention_finite3(value, parameters.value, linear))
        attention_fail(status, {nonfinite_value}u);
    if (parameters.keep_present != 0 && !attention_finite2(keep, parameters.keep, linear))
        attention_fail(status, {nonfinite_keep}u);

    const long long query_sequence = parameters.query.shape[1];
    const long long rows = parameters.query.shape[0] * query_sequence;
    if (linear >= rows) return;
    const long long batch = linear / query_sequence;
    const long long query_index = linear % query_sequence;
    const long long key_sequence = parameters.key.shape[1];
    {scalar} maximum = ({scalar})0;
    bool any = false;
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        if (!attention_active(keep, parameters, batch, query_index, key_index)) continue;
        const {scalar} score = attention_score(
            query, key, parameters, batch, query_index, key_index, scale
        );
        if (!isfinite(score)) attention_fail(status, {weights_arithmetic}u);
        maximum = !any || score > maximum ? score : maximum;
        any = true;
    }}
    if (!any) return;

    {scalar} denominator = ({scalar})0;
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        if (!attention_active(keep, parameters, batch, query_index, key_index)) continue;
        const {scalar} score = attention_score(
            query, key, parameters, batch, query_index, key_index, scale
        );
        const {scalar} weight = {exponential}(score - maximum);
        denominator += weight;
        if (!isfinite(weight) || !isfinite(denominator))
            attention_fail(status, {weights_arithmetic}u);
    }}
    if (!(denominator > ({scalar})0) || !isfinite(denominator)) {{
        attention_fail(status, {weights_arithmetic}u);
        return;
    }}

    for (long long feature = 0; feature < parameters.value.shape[2]; ++feature) {{
        {scalar} result = ({scalar})0;
        {scalar} total_weight = ({scalar})0;
        for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
            if (!attention_active(keep, parameters, batch, query_index, key_index)) continue;
            const {scalar} score = attention_score(
                query, key, parameters, batch, query_index, key_index, scale
            );
            const {scalar} weight = {exponential}(score - maximum) / denominator;
            if (!isfinite(weight)) {{
                attention_fail(status, {weights_arithmetic}u);
                continue;
            }}
            if (weight == ({scalar})0) continue;
            const {scalar} next_total = total_weight + weight;
            if (!isfinite(next_total) || !(next_total > ({scalar})0)) {{
                attention_fail(status, {weights_arithmetic}u);
                continue;
            }}
            result = attention_convex(
                result,
                value[physical3(parameters.value, batch, key_index, feature)],
                weight / next_total
            );
            if (!isfinite(result)) attention_fail(status, {output_arithmetic}u);
            total_weight = next_total;
        }}
    }}
}}
"#,
        prelude = prelude(),
        nonfinite_query = AttentionSemanticStatus::NonFiniteQuery.code(),
        nonfinite_key = AttentionSemanticStatus::NonFiniteKey.code(),
        nonfinite_value = AttentionSemanticStatus::NonFiniteValue.code(),
        nonfinite_keep = AttentionSemanticStatus::NonFiniteKeep.code(),
        weights_arithmetic = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code(),
        output_arithmetic = AttentionSemanticStatus::NonFiniteOutputArithmetic.code(),
    )
}

pub(crate) fn backward_validation_source(scalar: &str, epsilon: &str) -> String {
    format!(
        r#"{prelude}
extern "C" __device__ __forceinline__ bool attention_finite3(
    const {scalar}* values,
    const LayoutMeta layout,
    const long long linear
) {{
    const long long count = layout.shape[0] * layout.shape[1] * layout.shape[2];
    if (linear >= count) return true;
    const long long third = linear % layout.shape[2];
    const long long row = linear / layout.shape[2];
    const long long second = row % layout.shape[1];
    const long long first = row / layout.shape[1];
    return isfinite(values[physical3(layout, first, second, third)]);
}}

extern "C" __global__ void attention_backward_preflight(
    const {scalar}* grad_output,
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* weights,
    const {scalar}* query_gradient,
    const {scalar}* key_gradient,
    const {scalar}* value_gradient,
    unsigned int* status,
    const BackwardPreflightMeta parameters
) {{
    const long long linear = (long long)(blockIdx.x * blockDim.x + threadIdx.x);
    if (!attention_finite3(query, parameters.query, linear))
        attention_fail(status, {nonfinite_query}u);
    if (!attention_finite3(key, parameters.key, linear))
        attention_fail(status, {nonfinite_key}u);
    if (!attention_finite3(value, parameters.value, linear))
        attention_fail(status, {nonfinite_value}u);
    if (!attention_finite3(grad_output, parameters.grad_output, linear))
        attention_fail(status, {nonfinite_grad_output}u);
    if (!attention_finite3(weights, parameters.weights, linear))
        attention_fail(status, {nonfinite_weights}u);
    if (parameters.query_selected != 0 &&
        !attention_finite3(query_gradient, parameters.query_gradient, linear))
        attention_fail(status, {nonfinite_query_gradient}u);
    if (parameters.key_selected != 0 &&
        !attention_finite3(key_gradient, parameters.key_gradient, linear))
        attention_fail(status, {nonfinite_key_gradient}u);
    if (parameters.value_selected != 0 &&
        !attention_finite3(value_gradient, parameters.value_gradient, linear))
        attention_fail(status, {nonfinite_value_gradient}u);

    const long long key_sequence = parameters.weights.shape[2];
    const {scalar} tolerance = ({scalar})({epsilon}) * ({scalar})key_sequence * ({scalar})4;
    if (linear == 0 && (!isfinite(tolerance) || tolerance >= ({scalar})0.5))
        attention_fail(status, {weights_arithmetic}u);
    const long long rows = parameters.weights.shape[0] * parameters.weights.shape[1];
    if (linear >= rows) return;
    const long long query_index = linear % parameters.weights.shape[1];
    const long long batch = linear / parameters.weights.shape[1];
    {scalar} sum = ({scalar})0;
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        const {scalar} weight =
            weights[physical3(parameters.weights, batch, query_index, key_index)];
        if (weight < ({scalar})0 || weight > ({scalar})1)
            attention_fail(status, {invalid_weights}u);
        sum += weight;
    }}
    const {scalar} delta = sum - ({scalar})1;
    const {scalar} absolute_delta = delta < ({scalar})0 ? -delta : delta;
    if (sum != ({scalar})0 && absolute_delta > tolerance)
        attention_fail(status, {invalid_weights}u);
}}
"#,
        prelude = prelude(),
        nonfinite_query = AttentionSemanticStatus::NonFiniteQuery.code(),
        nonfinite_key = AttentionSemanticStatus::NonFiniteKey.code(),
        nonfinite_value = AttentionSemanticStatus::NonFiniteValue.code(),
        nonfinite_grad_output = AttentionSemanticStatus::NonFiniteOutputGradient.code(),
        nonfinite_weights = AttentionSemanticStatus::NonFiniteWeights.code(),
        invalid_weights = AttentionSemanticStatus::InvalidWeights.code(),
        nonfinite_query_gradient = AttentionSemanticStatus::NonFiniteQueryGradient.code(),
        nonfinite_key_gradient = AttentionSemanticStatus::NonFiniteKeyGradient.code(),
        nonfinite_value_gradient = AttentionSemanticStatus::NonFiniteValueGradient.code(),
        weights_arithmetic = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_partition_work_and_use_shared_status_codes_without_headers() {
        let forward = forward_preflight_source("float", "expf");
        let backward = backward_validation_source("double", "2.220446049250313e-16");
        assert!(forward.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(backward.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(forward.contains("attention_fail(status, 1u)"));
        assert!(backward.contains("attention_fail(status, 7u)"));
        assert!(forward.contains("atomicMin(status, code)"));
        assert!(!forward.contains("#include"));
        assert!(!backward.contains("#include"));
    }
}
