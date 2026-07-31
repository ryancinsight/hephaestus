use super::common::prelude;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::application::attention) enum GradientTarget {
    Query,
    Key,
    Value,
}

impl GradientTarget {
    pub(in crate::application::attention) const fn entry(self) -> &'static str {
        match self {
            Self::Query => "hephaestus_attention_backward_query",
            Self::Key => "hephaestus_attention_backward_key",
            Self::Value => "hephaestus_attention_backward_value",
        }
    }
}

pub(in crate::application::attention) fn backward_source(
    scalar: &str,
    target: GradientTarget,
) -> String {
    match target {
        GradientTarget::Query => query_source(scalar, target.entry()),
        GradientTarget::Key => key_source(scalar, target.entry()),
        GradientTarget::Value => value_source(scalar, target.entry()),
    }
}

fn score_gradient_helper(scalar: &str) -> String {
    format!(
        r#"
extern "C" __device__ __forceinline__ {scalar} score_gradient(
    const {scalar}* grad_output,
    const {scalar}* value,
    const {scalar}* weights,
    const int batch,
    const int query_index,
    const int key_index,
    const {scalar} scale,
    const AttentionMeta parameters
) {{
    const int value_feature = parameters.value.shape[2];
    const int key_sequence = parameters.key.shape[1];
    {scalar} selected = ({scalar})0;
    for (int feature = 0; feature < value_feature; ++feature) {{
        selected += grad_output[physical3(parameters.grad_output, batch, query_index, feature)] *
            value[physical3(parameters.value, batch, key_index, feature)];
    }}
    {scalar} expectation = ({scalar})0;
    for (int candidate = 0; candidate < key_sequence; ++candidate) {{
        {scalar} projection = ({scalar})0;
        for (int feature = 0; feature < value_feature; ++feature) {{
            projection += grad_output[physical3(parameters.grad_output, batch, query_index, feature)] *
                value[physical3(parameters.value, batch, candidate, feature)];
        }}
        expectation += weights[physical3(parameters.weights, batch, query_index, candidate)] *
            projection;
    }}
    return scale * weights[physical3(parameters.weights, batch, query_index, key_index)] *
        (selected - expectation);
}}
"#
    )
}

fn query_source(scalar: &str, entry: &str) -> String {
    format!(
        r#"{prelude}
{helper}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* weights,
    {scalar}* destination,
    {scalar} scale,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int query_sequence = parameters.query.shape[1];
    const int key_feature = parameters.query.shape[2];
    const int elements = parameters.query.shape[0] * query_sequence * key_feature;
    if (linear >= elements) return;
    const int feature = linear % key_feature;
    const int row = linear / key_feature;
    const int query_index = row % query_sequence;
    const int batch = row / query_sequence;
    {scalar} gradient = ({scalar})0;
    for (int key_index = 0; key_index < parameters.key.shape[1]; ++key_index) {{
        gradient += score_gradient(grad_output, value, weights, batch, query_index,
            key_index, scale, parameters) *
            key[physical3(parameters.key, batch, key_index, feature)];
    }}
    destination[physical3(parameters.destination, batch, query_index, feature)] += gradient;
}}
"#,
        prelude = prelude(),
        helper = score_gradient_helper(scalar),
    )
}

fn key_source(scalar: &str, entry: &str) -> String {
    format!(
        r#"{prelude}
{helper}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* weights,
    {scalar}* destination,
    {scalar} scale,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int key_sequence = parameters.key.shape[1];
    const int key_feature = parameters.key.shape[2];
    const int elements = parameters.key.shape[0] * key_sequence * key_feature;
    if (linear >= elements) return;
    const int feature = linear % key_feature;
    const int row = linear / key_feature;
    const int key_index = row % key_sequence;
    const int batch = row / key_sequence;
    {scalar} gradient = ({scalar})0;
    for (int query_index = 0; query_index < parameters.query.shape[1]; ++query_index) {{
        gradient += score_gradient(grad_output, value, weights, batch, query_index,
            key_index, scale, parameters) *
            query[physical3(parameters.query, batch, query_index, feature)];
    }}
    destination[physical3(parameters.destination, batch, key_index, feature)] += gradient;
}}
"#,
        prelude = prelude(),
        helper = score_gradient_helper(scalar),
    )
}

fn value_source(scalar: &str, entry: &str) -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* value,
    const {scalar}* weights,
    {scalar}* destination,
    {scalar} scale,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int key_sequence = parameters.value.shape[1];
    const int value_feature = parameters.value.shape[2];
    const int elements = parameters.value.shape[0] * key_sequence * value_feature;
    if (linear >= elements) return;
    const int feature = linear % value_feature;
    const int row = linear / value_feature;
    const int key_index = row % key_sequence;
    const int batch = row / key_sequence;
    {scalar} gradient = ({scalar})0;
    for (int query_index = 0; query_index < parameters.query.shape[1]; ++query_index) {{
        gradient += weights[physical3(parameters.weights, batch, query_index, key_index)] *
            grad_output[physical3(parameters.grad_output, batch, query_index, feature)];
    }}
    destination[physical3(parameters.destination, batch, key_index, feature)] += gradient;
}}
"#,
        prelude = prelude(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backward_target_is_additive_and_strided() {
        for target in [
            GradientTarget::Query,
            GradientTarget::Key,
            GradientTarget::Value,
        ] {
            let source = backward_source("double", target);
            assert!(source.contains("destination[physical3(parameters.destination"));
            assert!(source.contains("] += gradient"));
            assert!(source.contains(target.entry()));
        }
    }
}
