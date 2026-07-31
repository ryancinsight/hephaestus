use super::common::prelude;

pub(in crate::application::attention) const ENTRY: &str = "hephaestus_attention_forward";

fn helpers(scalar: &str) -> String {
    format!(
        r#"
extern "C" __device__ __forceinline__ {scalar} attention_score(
    const {scalar}* query,
    const {scalar}* key,
    const int batch,
    const int query_index,
    const int key_index,
    const AttentionMeta parameters
) {{
    {scalar} score = ({scalar})0;
    for (int feature = 0; feature < parameters.query.shape[2]; ++feature) {{
        score += query[physical3(parameters.query, batch, query_index, feature)] *
            key[physical3(parameters.key, batch, key_index, feature)];
    }}
    return score;
}}

extern "C" __device__ __forceinline__ {scalar} attention_convex(
    const {scalar} current,
    const {scalar} value,
    const {scalar} weight
) {{
    if ((current >= ({scalar})0) == (value >= ({scalar})0)) {{
        if (current <= value) {{
            return current + weight * (value - current);
        }}
        return value + (({scalar})1 - weight) * (current - value);
    }}
    return (({scalar})1 - weight) * current + weight * value;
}}
"#
    )
}

pub(in crate::application::attention) fn forward_source(scalar: &str) -> String {
    let exponential = if scalar == "float" { "expf" } else { "exp" };
    format!(
        r#"{prelude}
#include <math.h>
{helpers}

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
    const int rows = parameters.query.shape[0] * query_sequence;
    if (row >= rows) return;
    const int batch = row / query_sequence;
    const int query_index = row % query_sequence;
    const int key_sequence = parameters.key.shape[1];
    const int mask_batch = batch / parameters.heads_per_batch;

    {scalar} maximum = ({scalar})0;
    bool any = false;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        const bool kept = (!parameters.causal || key_index <= query_index) &&
            (!parameters.has_keep ||
             keep[physical2(parameters.keep, mask_batch, key_index)] != ({scalar})0);
        if (!kept) {{
            weights[physical3(parameters.weights, batch, query_index, key_index)] = ({scalar})0;
            continue;
        }}
        const {scalar} score = attention_score(
            query, key, batch, query_index, key_index, parameters) * scale;
        maximum = !any || score > maximum ? score : maximum;
        any = true;
    }}

    {scalar} denominator = ({scalar})0;
    if (any) {{
        for (int key_index = 0; key_index < key_sequence; ++key_index) {{
            const bool kept = (!parameters.causal || key_index <= query_index) &&
                (!parameters.has_keep ||
                 keep[physical2(parameters.keep, mask_batch, key_index)] != ({scalar})0);
            if (!kept) continue;
            const {scalar} score = attention_score(
                query, key, batch, query_index, key_index, parameters) * scale;
            const {scalar} weight = {exponential}(score - maximum);
            weights[physical3(parameters.weights, batch, query_index, key_index)] = weight;
            denominator += weight;
        }}
        for (int key_index = 0; key_index < key_sequence; ++key_index) {{
            const int position = physical3(parameters.weights, batch, query_index, key_index);
            weights[position] /= denominator;
        }}
    }}

    for (int feature = 0; feature < parameters.value.shape[2]; ++feature) {{
        {scalar} result = ({scalar})0;
        {scalar} total_weight = ({scalar})0;
        if (any) {{
            for (int key_index = 0; key_index < key_sequence; ++key_index) {{
                const {scalar} weight =
                    weights[physical3(parameters.weights, batch, query_index, key_index)];
                if (weight == ({scalar})0) continue;
                const {scalar} next_total = total_weight + weight;
                result = attention_convex(
                    result,
                    value[physical3(parameters.value, batch, key_index, feature)],
                    weight / next_total);
                total_weight = next_total;
            }}
        }}
        output[physical3(parameters.output, batch, query_index, feature)] = result;
    }}
}}
"#,
        prelude = prelude(),
        helpers = helpers(scalar),
        entry = ENTRY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_is_grouped_causal_and_stably_convex() {
        let source = forward_source("double");

        assert!(source.contains("batch / parameters.heads_per_batch"));
        assert!(source.contains("key_index <= query_index"));
        assert!(source.contains("attention_convex"));
        assert!(source.contains("if (weight == (double)0) continue"));
        assert!(source.contains("exp(score - maximum)"));
    }
}
