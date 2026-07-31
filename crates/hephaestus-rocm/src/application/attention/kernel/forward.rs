use super::common::prelude;

pub(in crate::application::attention) const ENTRY: &str = "hephaestus_attention_forward";

pub(in crate::application::attention) fn forward_source(scalar: &str) -> String {
    let exponential = if scalar == "float" { "expf" } else { "exp" };
    format!(
        r#"{prelude}
#include <math.h>

extern "C" __global__ void {entry}(
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* keep,
    {scalar}* output,
    {scalar}* weights,
    {scalar} scale,
    AttentionMeta parameters
) {{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int query_sequence = parameters.query.shape[1];
    const int row_count = parameters.query.shape[0] * query_sequence;
    if (row >= row_count) {{
        return;
    }}
    const int batch = row / query_sequence;
    const int query_index = row % query_sequence;
    const int key_sequence = parameters.key.shape[1];
    const int key_feature = parameters.query.shape[2];
    const int value_feature = parameters.value.shape[2];
    const int mask_batch = batch / parameters.heads_per_batch;

    {scalar} maximum = -INFINITY;
    bool any = false;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        const bool kept = (!parameters.causal || key_index <= query_index) &&
            (!parameters.has_keep ||
             keep[physical2(parameters.keep, mask_batch, key_index)] != ({scalar})0);
        if (!kept) {{
            weights[physical3(parameters.weights, batch, query_index, key_index)] = ({scalar})0;
            continue;
        }}
        {scalar} score = ({scalar})0;
        for (int feature = 0; feature < key_feature; ++feature) {{
            score += query[physical3(parameters.query, batch, query_index, feature)] *
                key[physical3(parameters.key, batch, key_index, feature)];
        }}
        score *= scale;
        maximum = any && maximum > score ? maximum : score;
        any = true;
    }}

    {scalar} denominator = ({scalar})0;
    if (any) {{
        for (int key_index = 0; key_index < key_sequence; ++key_index) {{
            const bool kept = (!parameters.causal || key_index <= query_index) &&
                (!parameters.has_keep ||
                 keep[physical2(parameters.keep, mask_batch, key_index)] != ({scalar})0);
            if (!kept) {{
                continue;
            }}
            {scalar} score = ({scalar})0;
            for (int feature = 0; feature < key_feature; ++feature) {{
                score += query[physical3(parameters.query, batch, query_index, feature)] *
                    key[physical3(parameters.key, batch, key_index, feature)];
            }}
            const {scalar} weight = {exponential}(score * scale - maximum);
            weights[physical3(parameters.weights, batch, query_index, key_index)] = weight;
            denominator += weight;
        }}
    }}

    if (denominator != ({scalar})0) {{
        for (int key_index = 0; key_index < key_sequence; ++key_index) {{
            const int weight_position =
                physical3(parameters.weights, batch, query_index, key_index);
            weights[weight_position] /= denominator;
        }}
    }}

    for (int feature = 0; feature < value_feature; ++feature) {{
        {scalar} result = ({scalar})0;
        if (denominator != ({scalar})0) {{
            for (int key_index = 0; key_index < key_sequence; ++key_index) {{
                const int weight_position =
                    physical3(parameters.weights, batch, query_index, key_index);
                result += weights[weight_position] *
                    value[physical3(parameters.value, batch, key_index, feature)];
            }}
        }}
        output[physical3(parameters.output, batch, query_index, feature)] = result;
    }}
}}
"#,
        prelude = prelude(),
        entry = ENTRY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_encodes_grouped_causal_and_fully_masked_contracts() {
        let source = forward_source("float");

        assert!(source.contains("batch / parameters.heads_per_batch"));
        assert!(source.contains("key_index <= query_index"));
        assert!(source.contains("if (any)"));
        assert!(source.contains("result = (float)0"));
        assert!(source.contains("physical3(parameters.output"));
    }
}
