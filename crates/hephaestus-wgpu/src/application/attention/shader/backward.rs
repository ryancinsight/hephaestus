use super::super::WgslAttentionScalar;
use super::prelude::prelude;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::application::attention) enum BackwardStage {
    Score,
    Query,
    Key,
    Value,
}

pub(in crate::application::attention) fn backward_shader<T: WgslAttentionScalar>(
    stage: BackwardStage,
    width: u32,
) -> String {
    let prelude = prelude::<T>(width);
    let (bindings, body) = match stage {
        BackwardStage::Score => (score_bindings::<T>(), score_body::<T>()),
        BackwardStage::Query => (query_bindings::<T>(), query_body::<T>()),
        BackwardStage::Key => (key_bindings::<T>(), key_body::<T>()),
        BackwardStage::Value => (value_bindings::<T>(), value_body::<T>()),
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

fn score_bindings<T: WgslAttentionScalar>() -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> grad_output: array<{ty}>;
@group(0) @binding(1) var<storage, read> value: array<{ty}>;
@group(0) @binding(2) var<storage, read> weights: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> score_gradient: array<{ty}>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;"#,
        ty = T::TYPE_TOKEN,
    )
}

fn query_bindings<T: WgslAttentionScalar>() -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> score_gradient: array<{ty}>;
@group(0) @binding(1) var<storage, read> key: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;"#,
        ty = T::TYPE_TOKEN,
    )
}

fn key_bindings<T: WgslAttentionScalar>() -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> score_gradient: array<{ty}>;
@group(0) @binding(1) var<storage, read> query: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;"#,
        ty = T::TYPE_TOKEN,
    )
}

fn value_bindings<T: WgslAttentionScalar>() -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> weights: array<{ty}>;
@group(0) @binding(1) var<storage, read> grad_output: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;"#,
        ty = T::TYPE_TOKEN,
    )
}

fn score_body<T: WgslAttentionScalar>() -> String {
    format!(
        r#"
    let elements = parameters.dimensions.x * parameters.dimensions.y * parameters.dimensions.z;
    if (id.x >= elements) {{ return; }}
    let key_index = id.x % parameters.dimensions.z;
    let row = id.x / parameters.dimensions.z;
    let query_index = row % parameters.dimensions.y;
    let batch = row / parameters.dimensions.y;
    var projection: {ty} = 0.0;
    var candidate = 0u;
    loop {{
        if (candidate >= parameters.dimensions.z) {{ break; }}
        var candidate_gradient: {ty} = 0.0;
        var feature = 0u;
        loop {{
            if (feature >= parameters.value_and_flags.x) {{ break; }}
            candidate_gradient += grad_output[
                physical(parameters.grad_output, batch, query_index, feature)
            ] * value[physical(parameters.value, batch, candidate, feature)];
            feature += 1u;
        }}
        projection += weights[physical(parameters.weights, batch, query_index, candidate)] *
            candidate_gradient;
        candidate += 1u;
    }}
    var current_gradient: {ty} = 0.0;
    var feature = 0u;
    loop {{
        if (feature >= parameters.value_and_flags.x) {{ break; }}
        current_gradient += grad_output[
            physical(parameters.grad_output, batch, query_index, feature)
        ] * value[physical(parameters.value, batch, key_index, feature)];
        feature += 1u;
    }}
    let weight = weights[physical(parameters.weights, batch, query_index, key_index)];
    score_gradient[id.x] = weight * (current_gradient - projection);
"#,
        ty = T::TYPE_TOKEN,
    )
}

fn query_body<T: WgslAttentionScalar>() -> String {
    format!(
        r#"
    let elements = parameters.dimensions.x * parameters.dimensions.y * parameters.dimensions.w;
    if (id.x >= elements) {{ return; }}
    let feature = id.x % parameters.dimensions.w;
    let row = id.x / parameters.dimensions.w;
    let query_index = row % parameters.dimensions.y;
    let batch = row / parameters.dimensions.y;
    var accumulated: {ty} = 0.0;
    var key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        let score_index = (batch * parameters.dimensions.y + query_index) *
            parameters.dimensions.z + key_index;
        accumulated += score_gradient[score_index] *
            key[physical(parameters.key, batch, key_index, feature)];
        key_index += 1u;
    }}
    let index = physical(parameters.destination, batch, query_index, feature);
    destination[index] += parameters.scale_and_padding.x * accumulated;
"#,
        ty = T::TYPE_TOKEN,
    )
}

fn key_body<T: WgslAttentionScalar>() -> String {
    format!(
        r#"
    let elements = parameters.dimensions.x * parameters.dimensions.z * parameters.dimensions.w;
    if (id.x >= elements) {{ return; }}
    let feature = id.x % parameters.dimensions.w;
    let row = id.x / parameters.dimensions.w;
    let key_index = row % parameters.dimensions.z;
    let batch = row / parameters.dimensions.z;
    var accumulated: {ty} = 0.0;
    var query_index = 0u;
    loop {{
        if (query_index >= parameters.dimensions.y) {{ break; }}
        let score_index = (batch * parameters.dimensions.y + query_index) *
            parameters.dimensions.z + key_index;
        accumulated += score_gradient[score_index] *
            query[physical(parameters.query, batch, query_index, feature)];
        query_index += 1u;
    }}
    let index = physical(parameters.destination, batch, key_index, feature);
    destination[index] += parameters.scale_and_padding.x * accumulated;
"#,
        ty = T::TYPE_TOKEN,
    )
}

fn value_body<T: WgslAttentionScalar>() -> String {
    format!(
        r#"
    let elements = parameters.dimensions.x * parameters.dimensions.z * parameters.value_and_flags.x;
    if (id.x >= elements) {{ return; }}
    let feature = id.x % parameters.value_and_flags.x;
    let row = id.x / parameters.value_and_flags.x;
    let key_index = row % parameters.dimensions.z;
    let batch = row / parameters.dimensions.z;
    var accumulated: {ty} = 0.0;
    var query_index = 0u;
    loop {{
        if (query_index >= parameters.dimensions.y) {{ break; }}
        accumulated += weights[physical(parameters.weights, batch, query_index, key_index)] *
            grad_output[physical(parameters.grad_output, batch, query_index, feature)];
        query_index += 1u;
    }}
    let index = physical(parameters.destination, batch, key_index, feature);
    destination[index] += accumulated;
"#,
        ty = T::TYPE_TOKEN,
    )
}
