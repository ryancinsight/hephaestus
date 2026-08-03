use hephaestus_core::AttentionSemanticStatus;

use super::super::prelude::prelude;

fn helpers(scalar: &str) -> String {
    format!(
        r#"
extern "C" __device__ __forceinline__ bool attention_kept(
    const {scalar}* keep,
    const int batch,
    const int query_index,
    const int key_index,
    const AttentionMeta parameters
) {{
    return (!parameters.causal || key_index <= query_index) &&
        (!parameters.has_keep || keep[physical2(
            parameters.keep, batch / parameters.heads_per_batch, key_index)] != ({scalar})0);
}}

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
        if (current <= value) return current + weight * (value - current);
        return value + (({scalar})1 - weight) * (current - value);
    }}
    return (({scalar})1 - weight) * current + weight * value;
}}
"#
    )
}

pub(in crate::application::attention) const ENTRY: &str =
    "hephaestus_attention_forward_arithmetic_preflight";

pub(in crate::application::attention) fn forward_arithmetic_source(scalar: &str) -> String {
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
    unsigned int* status,
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
    {scalar} maximum = ({scalar})0;
    bool any = false;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        if (!attention_kept(keep, batch, query_index, key_index, parameters)) continue;
        const {scalar} score = attention_score(
            query, key, batch, query_index, key_index, parameters) * scale;
        if (!isfinite(score)) atomicMin(status, {weights_failure}u);
        maximum = !any || score > maximum ? score : maximum;
        any = true;
    }}
    if (!any) return;

    {scalar} denominator = ({scalar})0;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        if (!attention_kept(keep, batch, query_index, key_index, parameters)) continue;
        const {scalar} score = attention_score(
            query, key, batch, query_index, key_index, parameters) * scale;
        const {scalar} weight = {exponential}(score - maximum);
        denominator += weight;
        if (!isfinite(weight) || !isfinite(denominator)) {{
            atomicMin(status, {weights_failure}u);
        }}
    }}
    if (!(denominator > ({scalar})0) || !isfinite(denominator)) {{
        atomicMin(status, {weights_failure}u);
        return;
    }}

    for (int feature = 0; feature < parameters.value.shape[2]; ++feature) {{
        {scalar} result = ({scalar})0;
        {scalar} total_weight = ({scalar})0;
        for (int key_index = 0; key_index < key_sequence; ++key_index) {{
            if (!attention_kept(keep, batch, query_index, key_index, parameters)) continue;
            const {scalar} score = attention_score(
                query, key, batch, query_index, key_index, parameters) * scale;
            const {scalar} weight = {exponential}(score - maximum) / denominator;
            if (weight == ({scalar})0) continue;
            const {scalar} next_total = total_weight + weight;
            if (!isfinite(weight) || !isfinite(next_total) || !(next_total > ({scalar})0)) {{
                atomicMin(status, {weights_failure}u);
                continue;
            }}
            result = attention_convex(
                result,
                value[physical3(parameters.value, batch, key_index, feature)],
                weight / next_total);
            if (!isfinite(result)) atomicMin(status, {output_failure}u);
            total_weight = next_total;
        }}
    }}
}}
"#,
        prelude = prelude(),
        helpers = helpers(scalar),
        entry = ENTRY,
        weights_failure = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code(),
        output_failure = AttentionSemanticStatus::NonFiniteOutputArithmetic.code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_is_row_parallel_and_skips_zero_probabilities() {
        let source = forward_arithmetic_source("float");

        assert!(source.contains("const int row ="));
        assert!(source.contains("if (weight == (float)0) continue"));
        assert!(source.contains("attention_convex"));
        assert!(!source.contains("for (int batch = 0"));
    }
}
