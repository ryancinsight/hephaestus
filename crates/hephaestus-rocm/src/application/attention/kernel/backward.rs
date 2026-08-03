use super::prelude::prelude;

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

    pub(in crate::application::attention) const fn preflight_entry(self) -> &'static str {
        match self {
            Self::Query => "hephaestus_attention_backward_query_preflight",
            Self::Key => "hephaestus_attention_backward_key_preflight",
            Self::Value => "hephaestus_attention_backward_value_preflight",
        }
    }
}

pub(in crate::application::attention) fn backward_source(
    scalar: &str,
    target: GradientTarget,
) -> String {
    let body = match target {
        GradientTarget::Query => query_body(),
        GradientTarget::Key => key_body(),
        GradientTarget::Value => value_body(),
    };
    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* weights,
    const {scalar}* score_gradient,
    {scalar}* destination,
    {scalar} scale,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    {body}
}}
"#,
        prelude = prelude(),
        entry = target.entry(),
    )
    .replace("SCALAR", scalar)
}

fn query_body() -> &'static str {
    r#"
    const int feature_extent = parameters.query.shape[2];
    const int query_sequence = parameters.query.shape[1];
    const int key_sequence = parameters.key.shape[1];
    const int elements = parameters.query.shape[0] * query_sequence * feature_extent;
    if (linear >= elements) return;
    const int feature = linear % feature_extent;
    const int row = linear / feature_extent;
    const int query_index = row % query_sequence;
    const int batch = row / query_sequence;
    SCALAR sum = (SCALAR)0;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {
        sum += score_gradient[row * key_sequence + key_index] *
            key[physical3(parameters.key, batch, key_index, feature)];
    }
    destination[physical3(parameters.destination, batch, query_index, feature)] += scale * sum;
"#
}

fn key_body() -> &'static str {
    r#"
    const int feature_extent = parameters.key.shape[2];
    const int query_sequence = parameters.query.shape[1];
    const int key_sequence = parameters.key.shape[1];
    const int elements = parameters.key.shape[0] * key_sequence * feature_extent;
    if (linear >= elements) return;
    const int feature = linear % feature_extent;
    const int row = linear / feature_extent;
    const int key_index = row % key_sequence;
    const int batch = row / key_sequence;
    SCALAR sum = (SCALAR)0;
    for (int query_index = 0; query_index < query_sequence; ++query_index) {
        sum += score_gradient[(batch * query_sequence + query_index) * key_sequence + key_index] *
            query[physical3(parameters.query, batch, query_index, feature)];
    }
    destination[physical3(parameters.destination, batch, key_index, feature)] += scale * sum;
"#
}

fn value_body() -> &'static str {
    r#"
    const int feature_extent = parameters.value.shape[2];
    const int key_sequence = parameters.value.shape[1];
    const int elements = parameters.value.shape[0] * key_sequence * feature_extent;
    if (linear >= elements) return;
    const int feature = linear % feature_extent;
    const int row = linear / feature_extent;
    const int key_index = row % key_sequence;
    const int batch = row / key_sequence;
    SCALAR increment = (SCALAR)0;
    for (int query_index = 0; query_index < parameters.query.shape[1]; ++query_index) {
        increment += weights[physical3(parameters.weights, batch, query_index, key_index)] *
            grad_output[physical3(parameters.grad_output, batch, query_index, feature)];
    }
    destination[physical3(parameters.destination, batch, key_index, feature)] += increment;
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_and_key_reuse_dense_score_workspace() {
        for target in [GradientTarget::Query, GradientTarget::Key] {
            let source = backward_source("float", target);
            assert!(source.contains("score_gradient["));
            assert!(source.contains("] += scale * sum"));
            assert!(!source.contains("attention_score_gradient("));
        }
    }

    #[test]
    fn every_target_is_additive_and_strided() {
        for target in [
            GradientTarget::Query,
            GradientTarget::Key,
            GradientTarget::Value,
        ] {
            let source = backward_source("double", target);
            assert!(source.contains("destination[physical3(parameters.destination"));
            assert!(source.contains("] +="));
            assert!(source.contains(target.entry()));
            assert!(!source.contains("SCALAR"));
        }
    }
}
