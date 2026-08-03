use super::prelude::prelude;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::application::attention) enum BackwardStage {
    Score,
    Query,
    Key,
    Value,
}

pub(in crate::application::attention) fn backward_shader(
    stage: BackwardStage,
    width: u32,
) -> String {
    let prelude = prelude(width);
    let (bindings, body) = match stage {
        BackwardStage::Score => (score_bindings(), score_body()),
        BackwardStage::Query => (query_bindings(), query_body()),
        BackwardStage::Key => (key_bindings(), key_body()),
        BackwardStage::Value => (value_bindings(), value_body()),
    };
    format!(
        r#"{prelude}
{bindings}
@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    {body}
}}
"#
    )
}

fn score_bindings() -> &'static str {
    r#"@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> score_gradient: array<f32>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;"#
}

fn query_bindings() -> &'static str {
    r#"@group(0) @binding(0) var<storage, read> score_gradient: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;"#
}

fn key_bindings() -> &'static str {
    r#"@group(0) @binding(0) var<storage, read> score_gradient: array<f32>;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;"#
}

fn value_bindings() -> &'static str {
    r#"@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;"#
}

fn score_body() -> &'static str {
    r#"
    let elements = parameters.dimensions.x * parameters.dimensions.y * parameters.dimensions.z;
    if (id.x >= elements) { return; }
    let key_index = id.x % parameters.dimensions.z;
    let row = id.x / parameters.dimensions.z;
    let query_index = row % parameters.dimensions.y;
    let batch = row / parameters.dimensions.y;
    var projection = 0.0;
    var candidate = 0u;
    loop {
        if (candidate >= parameters.dimensions.z) { break; }
        var candidate_gradient = 0.0;
        var feature = 0u;
        loop {
            if (feature >= parameters.value_and_flags.x) { break; }
            candidate_gradient += grad_output[
                physical(parameters.grad_output, batch, query_index, feature)
            ] * value[physical(parameters.value, batch, candidate, feature)];
            feature += 1u;
        }
        projection += weights[physical(parameters.weights, batch, query_index, candidate)] *
            candidate_gradient;
        candidate += 1u;
    }
    var current_gradient = 0.0;
    var feature = 0u;
    loop {
        if (feature >= parameters.value_and_flags.x) { break; }
        current_gradient += grad_output[
            physical(parameters.grad_output, batch, query_index, feature)
        ] * value[physical(parameters.value, batch, key_index, feature)];
        feature += 1u;
    }
    let weight = weights[physical(parameters.weights, batch, query_index, key_index)];
    score_gradient[id.x] = weight * (current_gradient - projection);
"#
}

fn query_body() -> &'static str {
    r#"
    let elements = parameters.dimensions.x * parameters.dimensions.y * parameters.dimensions.w;
    if (id.x >= elements) { return; }
    let feature = id.x % parameters.dimensions.w;
    let row = id.x / parameters.dimensions.w;
    let query_index = row % parameters.dimensions.y;
    let batch = row / parameters.dimensions.y;
    var accumulated = 0.0;
    var key_index = 0u;
    loop {
        if (key_index >= parameters.dimensions.z) { break; }
        let score_index = (batch * parameters.dimensions.y + query_index) *
            parameters.dimensions.z + key_index;
        accumulated += score_gradient[score_index] *
            key[physical(parameters.key, batch, key_index, feature)];
        key_index += 1u;
    }
    let index = physical(parameters.destination, batch, query_index, feature);
    destination[index] += parameters.scale_and_padding.x * accumulated;
"#
}

fn key_body() -> &'static str {
    r#"
    let elements = parameters.dimensions.x * parameters.dimensions.z * parameters.dimensions.w;
    if (id.x >= elements) { return; }
    let feature = id.x % parameters.dimensions.w;
    let row = id.x / parameters.dimensions.w;
    let key_index = row % parameters.dimensions.z;
    let batch = row / parameters.dimensions.z;
    var accumulated = 0.0;
    var query_index = 0u;
    loop {
        if (query_index >= parameters.dimensions.y) { break; }
        let score_index = (batch * parameters.dimensions.y + query_index) *
            parameters.dimensions.z + key_index;
        accumulated += score_gradient[score_index] *
            query[physical(parameters.query, batch, query_index, feature)];
        query_index += 1u;
    }
    let index = physical(parameters.destination, batch, key_index, feature);
    destination[index] += parameters.scale_and_padding.x * accumulated;
"#
}

fn value_body() -> &'static str {
    r#"
    let elements = parameters.dimensions.x * parameters.dimensions.z * parameters.value_and_flags.x;
    if (id.x >= elements) { return; }
    let feature = id.x % parameters.value_and_flags.x;
    let row = id.x / parameters.value_and_flags.x;
    let key_index = row % parameters.dimensions.z;
    let batch = row / parameters.dimensions.z;
    var accumulated = 0.0;
    var query_index = 0u;
    loop {
        if (query_index >= parameters.dimensions.y) { break; }
        accumulated += weights[physical(parameters.weights, batch, query_index, key_index)] *
            grad_output[physical(parameters.grad_output, batch, query_index, feature)];
        query_index += 1u;
    }
    let index = physical(parameters.destination, batch, key_index, feature);
    destination[index] += accumulated;
"#
}
