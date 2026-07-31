use super::GradientKernel;
use super::common::prelude;

pub(crate) fn score_gradient_source(scalar: &str) -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void attention_score_gradient(
    const {scalar}* grad_output,
    const {scalar}* value,
    const {scalar}* weights,
    {scalar}* score_gradient,
    const BackwardMeta parameters
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
    const long long value_feature = parameters.value.shape[2];
    {scalar} weighted_sum = ({scalar})0;
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        {scalar} probability_gradient = ({scalar})0;
        for (long long feature = 0; feature < value_feature; ++feature) {{
            probability_gradient +=
                grad_output[physical3(parameters.grad_output, batch, query_index, feature)] *
                value[physical3(parameters.value, batch, key_index, feature)];
        }}
        weighted_sum += probability_gradient *
            weights[physical3(parameters.weights, batch, query_index, key_index)];
    }}
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {{
        {scalar} probability_gradient = ({scalar})0;
        for (long long feature = 0; feature < value_feature; ++feature) {{
            probability_gradient +=
                grad_output[physical3(parameters.grad_output, batch, query_index, feature)] *
                value[physical3(parameters.value, batch, key_index, feature)];
        }}
        const {scalar} probability =
            weights[physical3(parameters.weights, batch, query_index, key_index)];
        score_gradient[row * key_sequence + key_index] =
            probability * (probability_gradient - weighted_sum);
    }}
}}
"#,
        prelude = prelude(),
    )
}

pub(crate) fn backward_source(scalar: &str, kernel: GradientKernel) -> String {
    let body = match kernel {
        GradientKernel::Query => query_body(),
        GradientKernel::Key => key_body(),
        GradientKernel::Value => value_body(),
    };
    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* query,
    const {scalar}* key,
    const {scalar}* weights,
    const {scalar}* score_gradient,
    {scalar}* target,
    const {scalar} scale,
    const BackwardMeta parameters
) {{
    const long long linear = (long long)(blockIdx.x * blockDim.x + threadIdx.x);
    {body}
}}
"#,
        prelude = prelude(),
        entry = kernel.entry(),
    )
    .replace("SCALAR", scalar)
}

fn query_body() -> &'static str {
    r#"
    const long long feature_extent = parameters.query.shape[2];
    const long long query_sequence = parameters.query.shape[1];
    const long long elements = parameters.query.shape[0] * query_sequence * feature_extent;
    if (linear >= elements) return;
    const long long feature = linear % feature_extent;
    const long long row = linear / feature_extent;
    const long long query_index = row % query_sequence;
    const long long batch = row / query_sequence;
    const long long key_sequence = parameters.key.shape[1];
    SCALAR sum = (SCALAR)0;
    for (long long key_index = 0; key_index < key_sequence; ++key_index) {
        sum += score_gradient[row * key_sequence + key_index] *
            key[physical3(parameters.key, batch, key_index, feature)];
    }
    target[physical3(parameters.target, batch, query_index, feature)] += scale * sum;
"#
}

fn key_body() -> &'static str {
    r#"
    const long long feature_extent = parameters.key.shape[2];
    const long long key_sequence = parameters.key.shape[1];
    const long long elements = parameters.key.shape[0] * key_sequence * feature_extent;
    if (linear >= elements) return;
    const long long feature = linear % feature_extent;
    const long long row = linear / feature_extent;
    const long long key_index = row % key_sequence;
    const long long batch = row / key_sequence;
    const long long query_sequence = parameters.query.shape[1];
    SCALAR sum = (SCALAR)0;
    for (long long query_index = 0; query_index < query_sequence; ++query_index) {
        sum += score_gradient[(batch * query_sequence + query_index) * key_sequence + key_index] *
            query[physical3(parameters.query, batch, query_index, feature)];
    }
    target[physical3(parameters.target, batch, key_index, feature)] += scale * sum;
"#
}

fn value_body() -> &'static str {
    r#"
    const long long feature_extent = parameters.value.shape[2];
    const long long key_sequence = parameters.value.shape[1];
    const long long elements = parameters.value.shape[0] * key_sequence * feature_extent;
    if (linear >= elements) return;
    const long long feature = linear % feature_extent;
    const long long row = linear / feature_extent;
    const long long key_index = row % key_sequence;
    const long long batch = row / key_sequence;
    const long long query_sequence = parameters.query.shape[1];
    SCALAR sum = (SCALAR)0;
    for (long long query_index = 0; query_index < query_sequence; ++query_index) {
        sum += weights[physical3(parameters.weights, batch, query_index, key_index)] *
            grad_output[physical3(parameters.grad_output, batch, query_index, feature)];
    }
    target[physical3(parameters.target, batch, key_index, feature)] += sum;
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_source_forms_softmax_jacobian_product() {
        let source = score_gradient_source("double");
        assert!(source.contains("probability * (probability_gradient - weighted_sum)"));
        assert!(source.contains("row * key_sequence + key_index"));
    }

    #[test]
    fn every_gradient_source_is_additive() {
        for kernel in [
            GradientKernel::Query,
            GradientKernel::Key,
            GradientKernel::Value,
        ] {
            let source = backward_source("float", kernel);
            assert!(source.contains("target[physical3(parameters.target"));
            assert!(source.contains("] +="));
            assert!(!source.contains("SCALAR"));
        }
    }
}
