use super::prelude::prelude;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::application::attention) enum ForwardStage {
    Weights,
    Output,
}

pub(in crate::application::attention) fn forward_shader(stage: ForwardStage, width: u32) -> String {
    match stage {
        ForwardStage::Weights => weights_shader(width),
        ForwardStage::Output => output_shader(width),
    }
}

fn weights_shader(width: u32) -> String {
    let prelude = prelude(width);
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> keep_mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;

fn is_kept(batch: u32, query_index: u32, key_index: u32) -> bool {{
    if (parameters.value_and_flags.y != 0u && key_index > query_index) {{ return false; }}
    if (parameters.value_and_flags.z == 0u) {{ return true; }}
    let mask_batch = batch / parameters.value_and_flags.w;
    return keep_mask[physical(parameters.keep_mask, mask_batch, key_index, 0u)] != 0.0;
}}

fn score(batch: u32, query_index: u32, key_index: u32) -> f32 {{
    var dot = 0.0;
    var feature = 0u;
    loop {{
        if (feature >= parameters.dimensions.w) {{ break; }}
        dot += query[physical(parameters.query, batch, query_index, feature)] *
            key[physical(parameters.key, batch, key_index, feature)];
        feature += 1u;
    }}
    return dot * parameters.scale_and_padding.x;
}}

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let rows = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= rows) {{ return; }}
    let batch = id.x / parameters.dimensions.y;
    let query_index = id.x % parameters.dimensions.y;
    var maximum = -3.402823466e+38;
    var kept_count = 0u;
    var key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        if (is_kept(batch, query_index, key_index)) {{
            maximum = max(maximum, score(batch, query_index, key_index));
            kept_count += 1u;
        }}
        key_index += 1u;
    }}
    var denominator = 0.0;
    key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        let index = physical(parameters.weights, batch, query_index, key_index);
        if (kept_count != 0u && is_kept(batch, query_index, key_index)) {{
            let numerator = exp(score(batch, query_index, key_index) - maximum);
            weights[index] = numerator;
            denominator += numerator;
        }} else {{
            weights[index] = 0.0;
        }}
        key_index += 1u;
    }}
    if (kept_count == 0u) {{ return; }}
    key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        let index = physical(parameters.weights, batch, query_index, key_index);
        weights[index] /= denominator;
        key_index += 1u;
    }}
}}
"#
    )
}

fn output_shader(width: u32) -> String {
    let prelude = prelude(width);
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: AttentionMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = parameters.dimensions.x * parameters.dimensions.y * parameters.value_and_flags.x;
    if (id.x >= elements) {{ return; }}
    let feature = id.x % parameters.value_and_flags.x;
    let row = id.x / parameters.value_and_flags.x;
    let query_index = row % parameters.dimensions.y;
    let batch = row / parameters.dimensions.y;
    var accumulated = 0.0;
    var total_weight = 0.0;
    var key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        let weight = weights[physical(parameters.weights, batch, query_index, key_index)];
        if (weight != 0.0) {{
            let next_total = total_weight + weight;
            let next = value[physical(parameters.value, batch, key_index, feature)];
            let fraction = weight / next_total;
            if ((accumulated >= 0.0) == (next >= 0.0)) {{
                if (accumulated <= next) {{
                    accumulated = accumulated + fraction * (next - accumulated);
                }} else {{
                    accumulated = next + (1.0 - fraction) * (accumulated - next);
                }}
            }} else {{
                accumulated = (1.0 - fraction) * accumulated + fraction * next;
            }}
            total_weight = next_total;
        }}
        key_index += 1u;
    }}
    output[physical(parameters.destination, batch, query_index, feature)] = accumulated;
}}
"#
    )
}
