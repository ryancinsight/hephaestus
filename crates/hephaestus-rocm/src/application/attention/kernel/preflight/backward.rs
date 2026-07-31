use hephaestus_core::AttentionSemanticStatus;

use super::super::GradientTarget;
use super::super::common::prelude;

pub(in crate::application::attention) const PROBABILITY_ENTRY: &str =
    "hephaestus_attention_backward_probability_preflight";

pub(in crate::application::attention) fn probability_source(scalar: &str) -> String {
    let absolute = if scalar == "float" { "fabsf" } else { "fabs" };
    let epsilon = if scalar == "float" {
        "1.1920928955078125e-7f"
    } else {
        "2.220446049250313080847263336181640625e-16"
    };
    format!(
        r#"{prelude}
#include <math.h>
extern "C" __global__ void {entry}(
    const {scalar}* weights,
    unsigned int* status,
    AttentionMeta parameters
) {{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int query_sequence = parameters.weights.shape[1];
    const int rows = parameters.weights.shape[0] * query_sequence;
    if (row >= rows) return;
    const int batch = row / query_sequence;
    const int query_index = row % query_sequence;
    const int key_sequence = parameters.weights.shape[2];
    const {scalar} tolerance = ({scalar})({epsilon}) * ({scalar})key_sequence * ({scalar})4;
    if (!isfinite(tolerance) || tolerance >= ({scalar})0.5) {{
        atomicMin(status, {arithmetic_failure}u);
        return;
    }}
    {scalar} sum = ({scalar})0;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        const {scalar} weight =
            weights[physical3(parameters.weights, batch, query_index, key_index)];
        if (weight < ({scalar})0 || weight > ({scalar})1) {{
            atomicMin(status, {invalid_failure}u);
        }}
        sum += weight;
    }}
    if (sum != ({scalar})0 && {absolute}(sum - ({scalar})1) > tolerance) {{
        atomicMin(status, {invalid_failure}u);
    }}
}}
"#,
        prelude = prelude(),
        entry = PROBABILITY_ENTRY,
        arithmetic_failure = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code(),
        invalid_failure = AttentionSemanticStatus::InvalidWeights.code(),
    )
}

pub(in crate::application::attention) const CANDIDATE_ENTRY: &str =
    "hephaestus_attention_backward_candidate_preflight";
pub(in crate::application::attention) const SCORE_ENTRY: &str =
    "hephaestus_attention_backward_score_preflight";

pub(in crate::application::attention) fn candidate_source(scalar: &str) -> String {
    format!(
        r#"{prelude}
#include <math.h>
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* value,
    {scalar}* candidate,
    unsigned int* status,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int query_sequence = parameters.query.shape[1];
    const int key_sequence = parameters.key.shape[1];
    const int elements = parameters.query.shape[0] * query_sequence * key_sequence;
    if (linear >= elements) return;
    const int key_index = linear % key_sequence;
    const int row = linear / key_sequence;
    const int query_index = row % query_sequence;
    const int batch = row / query_sequence;
    {scalar} result = ({scalar})0;
    for (int feature = 0; feature < parameters.value.shape[2]; ++feature) {{
        result += grad_output[physical3(parameters.grad_output, batch, query_index, feature)] *
            value[physical3(parameters.value, batch, key_index, feature)];
        if (!isfinite(result)) atomicMin(status, {failure}u);
    }}
    candidate[linear] = result;
}}
"#,
        prelude = prelude(),
        entry = CANDIDATE_ENTRY,
        failure = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code(),
    )
}

pub(in crate::application::attention) fn score_source(scalar: &str) -> String {
    format!(
        r#"{prelude}
#include <math.h>
extern "C" __global__ void {entry}(
    const {scalar}* candidate,
    const {scalar}* weights,
    {scalar}* score_gradient,
    unsigned int* status,
    AttentionMeta parameters
) {{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int query_sequence = parameters.query.shape[1];
    const int rows = parameters.query.shape[0] * query_sequence;
    if (row >= rows) return;
    const int batch = row / query_sequence;
    const int query_index = row % query_sequence;
    const int key_sequence = parameters.key.shape[1];
    {scalar} projection = ({scalar})0;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        projection += weights[physical3(parameters.weights, batch, query_index, key_index)] *
            candidate[row * key_sequence + key_index];
        if (!isfinite(projection)) atomicMin(status, {failure}u);
    }}
    for (int key_index = 0; key_index < key_sequence; ++key_index) {{
        const int index = row * key_sequence + key_index;
        const {scalar} result = weights[
            physical3(parameters.weights, batch, query_index, key_index)] *
            (candidate[index] - projection);
        if (!isfinite(result)) atomicMin(status, {failure}u);
        score_gradient[index] = result;
    }}
}}
"#,
        prelude = prelude(),
        entry = SCORE_ENTRY,
        failure = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code(),
    )
}

pub(in crate::application::attention) fn gradient_preflight_source(
    scalar: &str,
    target: GradientTarget,
) -> String {
    let (entry, body, destination_failure, arithmetic_failure) = match target {
        GradientTarget::Query => (
            target.preflight_entry(),
            query_body(),
            AttentionSemanticStatus::NonFiniteQueryGradient,
            AttentionSemanticStatus::NonFiniteQueryGradientArithmetic,
        ),
        GradientTarget::Key => (
            target.preflight_entry(),
            key_body(),
            AttentionSemanticStatus::NonFiniteKeyGradient,
            AttentionSemanticStatus::NonFiniteKeyGradientArithmetic,
        ),
        GradientTarget::Value => (
            target.preflight_entry(),
            value_body(),
            AttentionSemanticStatus::NonFiniteValueGradient,
            AttentionSemanticStatus::NonFiniteValueGradientArithmetic,
        ),
    };
    format!(
        r#"{prelude}
#include <math.h>
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* weights,
    const {scalar}* score_gradient,
    const {scalar}* source,
    const {scalar}* destination,
    unsigned int* status,
    {scalar} scale,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    {body}
    if (!isfinite(increment)) atomicMin(status, {arithmetic_failure}u);
    const int position = physical3(parameters.destination, batch, sequence, feature);
    const {scalar} current = destination[position];
    if (!isfinite(current)) atomicMin(status, {destination_failure}u);
    if (!isfinite(current + increment)) atomicMin(status, {arithmetic_failure}u);
}}
"#,
        prelude = prelude(),
        destination_failure = destination_failure.code(),
        arithmetic_failure = arithmetic_failure.code(),
    )
    .replace("SCALAR", scalar)
}

fn query_body() -> &'static str {
    r#"
    const int query_sequence = parameters.query.shape[1];
    const int key_sequence = parameters.key.shape[1];
    const int feature_extent = parameters.query.shape[2];
    const int elements = parameters.query.shape[0] * query_sequence * feature_extent;
    if (linear >= elements) return;
    const int feature = linear % feature_extent;
    const int row = linear / feature_extent;
    const int sequence = row % query_sequence;
    const int batch = row / query_sequence;
    SCALAR sum = (SCALAR)0;
    for (int key_index = 0; key_index < key_sequence; ++key_index) {
        sum += score_gradient[row * key_sequence + key_index] *
            source[physical3(parameters.key, batch, key_index, feature)];
    }
    const SCALAR increment = scale * sum;
"#
}

fn key_body() -> &'static str {
    r#"
    const int query_sequence = parameters.query.shape[1];
    const int key_sequence = parameters.key.shape[1];
    const int feature_extent = parameters.key.shape[2];
    const int elements = parameters.key.shape[0] * key_sequence * feature_extent;
    if (linear >= elements) return;
    const int feature = linear % feature_extent;
    const int row = linear / feature_extent;
    const int sequence = row % key_sequence;
    const int batch = row / key_sequence;
    SCALAR sum = (SCALAR)0;
    for (int query_index = 0; query_index < query_sequence; ++query_index) {
        sum += score_gradient[(batch * query_sequence + query_index) * key_sequence + sequence] *
            source[physical3(parameters.query, batch, query_index, feature)];
    }
    const SCALAR increment = scale * sum;
"#
}

fn value_body() -> &'static str {
    r#"
    const int key_sequence = parameters.value.shape[1];
    const int feature_extent = parameters.value.shape[2];
    const int elements = parameters.value.shape[0] * key_sequence * feature_extent;
    if (linear >= elements) return;
    const int feature = linear % feature_extent;
    const int row = linear / feature_extent;
    const int sequence = row % key_sequence;
    const int batch = row / key_sequence;
    SCALAR increment = (SCALAR)0;
    for (int query_index = 0; query_index < parameters.query.shape[1]; ++query_index) {
        increment += weights[physical3(parameters.weights, batch, query_index, sequence)] *
            grad_output[physical3(parameters.grad_output, batch, query_index, feature)];
    }
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materializes_score_workspace_once() {
        let candidate = candidate_source("double");
        let score = score_source("double");
        let query = gradient_preflight_source("double", GradientTarget::Query);

        assert!(candidate.contains("candidate[linear] = result"));
        assert!(score.contains("score_gradient[index] = result"));
        assert!(query.contains("score_gradient[row * key_sequence + key_index]"));
        assert!(!query.contains("candidate_gradient"));
    }
}
