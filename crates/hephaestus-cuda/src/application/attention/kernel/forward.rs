use super::common::prelude;

pub(crate) fn forward_source(scalar: &str, exponential: &str) -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void attention_forward(
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* keep,
    {scalar}* output,
    {scalar}* weights,
    const {scalar} scale,
    const ForwardMeta parameters
) {{
    const long long row = (long long)(blockIdx.x * blockDim.x + threadIdx.x);
    const long long query_sequence = parameters.query.shape[1];
    const long long rows = parameters.query.shape[0] * query_sequence;
    if (row >= rows) {{
        return;
    }}
    const long long batch = row / query_sequence;
    const long long query_index = row % query_sequence;
    const long long key_sequence = parameters.key.shape[1];
    const long long key_feature = parameters.query.shape[2];
    const long long value_feature = parameters.value.shape[2];
    {scalar} maximum = ({scalar})0;
    bool any = false;

    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        const bool causal_keep = parameters.causal == 0 || key_index <= query_index;
        const bool grouped_keep = parameters.keep_present == 0 ||
            keep[physical2(parameters.keep, batch / parameters.heads_per_batch, key_index)] != ({scalar})0;
        if (!causal_keep || !grouped_keep) {{
            continue;
        }}
        {scalar} score = ({scalar})0;
        for (long long feature = 0; feature < key_feature; ++feature) {{
            score += query[physical3(parameters.query, batch, query_index, feature)] *
                key[physical3(parameters.key, batch, key_index, feature)];
        }}
        score *= scale;
        maximum = !any || score > maximum ? score : maximum;
        any = true;
    }}

    if (!any) {{
        for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
            weights[physical3(parameters.weights, batch, query_index, key_index)] = ({scalar})0;
        }}
        for (long long feature = 0; feature < value_feature; ++feature) {{
            output[physical3(parameters.output, batch, query_index, feature)] = ({scalar})0;
        }}
        return;
    }}

    {scalar} denominator = ({scalar})0;
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        const bool causal_keep = parameters.causal == 0 || key_index <= query_index;
        const bool grouped_keep = parameters.keep_present == 0 ||
            keep[physical2(parameters.keep, batch / parameters.heads_per_batch, key_index)] != ({scalar})0;
        {scalar} weight = ({scalar})0;
        if (causal_keep && grouped_keep) {{
            {scalar} score = ({scalar})0;
            for (long long feature = 0; feature < key_feature; ++feature) {{
                score += query[physical3(parameters.query, batch, query_index, feature)] *
                    key[physical3(parameters.key, batch, key_index, feature)];
            }}
            weight = {exponential}(score * scale - maximum);
            denominator += weight;
        }}
        weights[physical3(parameters.weights, batch, query_index, key_index)] = weight;
    }}

    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        const long long weight_offset =
            physical3(parameters.weights, batch, query_index, key_index);
        weights[weight_offset] /= denominator;
    }}
    for (long long feature = 0; feature < value_feature; ++feature) {{
        {scalar} result = ({scalar})0;
        for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
            const long long weight_offset =
                physical3(parameters.weights, batch, query_index, key_index);
            result += weights[weight_offset] *
                value[physical3(parameters.value, batch, key_index, feature)];
        }}
        output[physical3(parameters.output, batch, query_index, feature)] = result;
    }}
}}
"#,
        prelude = prelude(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_covers_grouped_causal_and_fully_masked_rows() {
        let source = forward_source("float", "expf");
        assert!(source.contains("batch / parameters.heads_per_batch"));
        assert!(source.contains("key_index <= query_index"));
        assert!(source.contains("if (!any)"));
        assert!(source.contains("= (float)0;"));
        assert!(source.contains("expf(score * scale - maximum)"));
    }
}
