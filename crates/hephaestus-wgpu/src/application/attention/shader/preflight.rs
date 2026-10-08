use super::super::WgslAttentionScalar;
use super::prelude::prelude;
use hephaestus_core::AttentionSemanticStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::application::attention) enum GradientPreflightStage {
    Query,
    Key,
    Value,
}

pub(in crate::application::attention) fn finite_preflight_shader<T: WgslAttentionScalar>(
    layout: &'static str,
    rank: u32,
    failure: AttentionSemanticStatus,
    width: u32,
) -> String {
    let prelude = prelude::<T>(width);
    let third = if rank == 2 { "0u" } else { "third" };
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> source: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<uniform> parameters: AttentionMeta;

fn finite(value: {ty}) -> bool {{
    return value == value && abs(value) <= {finite_max};
}}

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let view_meta = parameters.{layout};
    let plane = view_meta.shape.y * view_meta.shape.z;
    let elements = view_meta.shape.x * plane;
    if (id.x >= elements) {{ return; }}
    let first = id.x / plane;
    let within = id.x % plane;
    let second = within / view_meta.shape.z;
    let third = within % view_meta.shape.z;
    if (!finite(source[physical(view_meta, first, second, {third})])) {{
        atomicMin(&status, {failure}u);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        finite_max = T::FINITE_MAX,
        failure = failure.code()
    )
}

pub(in crate::application::attention) fn linear_finite_preflight_shader<T: WgslAttentionScalar>(
    failure: AttentionSemanticStatus,
    width: u32,
) -> String {
    let prelude = prelude::<T>(width);
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> source: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<uniform> parameters: AttentionMeta;

fn finite(value: {ty}) -> bool {{
    return value == value && abs(value) <= {finite_max};
}}

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = parameters.dimensions.x * parameters.dimensions.y * parameters.dimensions.z;
    if (id.x < elements && !finite(source[id.x])) {{
        atomicMin(&status, {failure}u);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        finite_max = T::FINITE_MAX,
        failure = failure.code()
    )
}

pub(in crate::application::attention) fn forward_arithmetic_preflight_shader<
    T: WgslAttentionScalar,
>(
    width: u32,
) -> String {
    let prelude = prelude::<T>(width);
    let weight_failure = AttentionSemanticStatus::NonFiniteWeightsArithmetic.code();
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> query: array<{ty}>;
@group(0) @binding(1) var<storage, read> key: array<{ty}>;
@group(0) @binding(2) var<storage, read> keep_mask: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;

fn finite(number: {ty}) -> bool {{
    return number == number && abs(number) <= {finite_max};
}}

fn kept(batch: u32, query_index: u32, key_index: u32) -> bool {{
    if (parameters.value_and_flags.y != 0u && key_index > query_index) {{ return false; }}
    if (parameters.value_and_flags.z == 0u) {{ return true; }}
    let mask_batch = batch / parameters.value_and_flags.w;
    return keep_mask[physical(parameters.keep_mask, mask_batch, key_index, 0u)] != 0.0;
}}

fn attention_score(batch: u32, query_index: u32, key_index: u32) -> {ty} {{
    var dot: {ty} = 0.0;
    var feature = 0u;
    loop {{
        if (feature >= parameters.dimensions.w) {{ break; }}
        dot += query[physical(parameters.query, batch, query_index, feature)] *
            key[physical(parameters.key, batch, key_index, feature)];
        if (!finite(dot)) {{ atomicMin(&status, {weight_failure}u); }}
        feature += 1u;
    }}
    let scaled = dot * parameters.scale_and_padding.x;
    if (!finite(scaled)) {{ atomicMin(&status, {weight_failure}u); }}
    return scaled;
}}

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let rows = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= rows) {{ return; }}
    let batch = id.x / parameters.dimensions.y;
    let query_index = id.x % parameters.dimensions.y;
    var maximum: {ty} = {neg_max};
    var kept_count = 0u;
    var key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        if (kept(batch, query_index, key_index)) {{
            maximum = max(maximum, attention_score(batch, query_index, key_index));
            kept_count += 1u;
        }}
        key_index += 1u;
    }}
    if (kept_count == 0u) {{ return; }}
    var denominator: {ty} = 0.0;
    key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        if (kept(batch, query_index, key_index)) {{
            denominator += exp(attention_score(batch, query_index, key_index) - maximum);
            if (!finite(denominator)) {{ atomicMin(&status, {weight_failure}u); }}
        }}
        key_index += 1u;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        finite_max = T::FINITE_MAX,
        neg_max = T::NEG_MAX,
    )
}

pub(in crate::application::attention) fn backward_probability_preflight_shader<
    T: WgslAttentionScalar,
>(
    width: u32,
) -> String {
    let prelude = prelude::<T>(width);
    let failure = AttentionSemanticStatus::InvalidWeights.code();
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> weights: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<uniform> parameters: AttentionMeta;

fn finite(number: {ty}) -> bool {{
    return number == number && abs(number) <= {finite_max};
}}

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let rows = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= rows) {{ return; }}
    let batch = id.x / parameters.dimensions.y;
    let query_index = id.x % parameters.dimensions.y;
    var sum: {ty} = 0.0;
    var key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        let weight = weights[physical(parameters.weights, batch, query_index, key_index)];
        if (weight < 0.0 || weight > 1.0) {{ atomicMin(&status, {failure}u); }}
        sum += weight;
        key_index += 1u;
    }}
    let tolerance = 4.0 * {epsilon} * {ty}(parameters.dimensions.z);
    if (!finite(tolerance) || tolerance >= 0.5 ||
        (sum != 0.0 && abs(sum - 1.0) > tolerance)) {{
        atomicMin(&status, {failure}u);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        finite_max = T::FINITE_MAX,
        epsilon = T::EPSILON,
    )
}

pub(in crate::application::attention) fn backward_gradient_preflight_shader<
    T: WgslAttentionScalar,
>(
    stage: GradientPreflightStage,
    width: u32,
) -> String {
    let prelude = prelude::<T>(width);
    let (bindings, elements, coordinates, increment, destination_failure, arithmetic_failure) =
        match stage {
            GradientPreflightStage::Query => (
                format!(
                    r#"@group(0) @binding(0) var<storage, read> score_gradient: array<{ty}>;
@group(0) @binding(1) var<storage, read> source: array<{ty}>;
@group(0) @binding(2) var<storage, read> destination: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;"#,
                    ty = T::TYPE_TOKEN,
                ),
                "parameters.dimensions.x * parameters.dimensions.y * parameters.dimensions.w",
                r#"let feature = id.x % parameters.dimensions.w;
    let row = id.x / parameters.dimensions.w;
    let sequence = row % parameters.dimensions.y;
    let batch = row / parameters.dimensions.y;"#,
                format!(
                    r#"var increment: {ty} = 0.0;
    var key_index = 0u;
    loop {{
        if (key_index >= parameters.dimensions.z) {{ break; }}
        let score_index = (batch * parameters.dimensions.y + sequence) *
            parameters.dimensions.z + key_index;
        increment += score_gradient[score_index] *
            source[physical(parameters.key, batch, key_index, feature)];
        key_index += 1u;
    }}
    increment *= parameters.scale_and_padding.x;"#,
                    ty = T::TYPE_TOKEN,
                ),
                AttentionSemanticStatus::NonFiniteQueryGradient.code(),
                AttentionSemanticStatus::NonFiniteQueryGradientArithmetic.code(),
            ),
            GradientPreflightStage::Key => (
                format!(
                    r#"@group(0) @binding(0) var<storage, read> score_gradient: array<{ty}>;
@group(0) @binding(1) var<storage, read> source: array<{ty}>;
@group(0) @binding(2) var<storage, read> destination: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;"#,
                    ty = T::TYPE_TOKEN,
                ),
                "parameters.dimensions.x * parameters.dimensions.z * parameters.dimensions.w",
                r#"let feature = id.x % parameters.dimensions.w;
    let row = id.x / parameters.dimensions.w;
    let sequence = row % parameters.dimensions.z;
    let batch = row / parameters.dimensions.z;"#,
                format!(
                    r#"var increment: {ty} = 0.0;
    var query_index = 0u;
    loop {{
        if (query_index >= parameters.dimensions.y) {{ break; }}
        let score_index = (batch * parameters.dimensions.y + query_index) *
            parameters.dimensions.z + sequence;
        increment += score_gradient[score_index] *
            source[physical(parameters.query, batch, query_index, feature)];
        query_index += 1u;
    }}
    increment *= parameters.scale_and_padding.x;"#,
                    ty = T::TYPE_TOKEN,
                ),
                AttentionSemanticStatus::NonFiniteKeyGradient.code(),
                AttentionSemanticStatus::NonFiniteKeyGradientArithmetic.code(),
            ),
            GradientPreflightStage::Value => (
                format!(
                    r#"@group(0) @binding(0) var<storage, read> weights: array<{ty}>;
@group(0) @binding(1) var<storage, read> grad_output: array<{ty}>;
@group(0) @binding(2) var<storage, read> destination: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> parameters: AttentionMeta;"#,
                    ty = T::TYPE_TOKEN,
                ),
                "parameters.dimensions.x * parameters.dimensions.z * parameters.value_and_flags.x",
                r#"let feature = id.x % parameters.value_and_flags.x;
    let row = id.x / parameters.value_and_flags.x;
    let sequence = row % parameters.dimensions.z;
    let batch = row / parameters.dimensions.z;"#,
                format!(
                    r#"var increment: {ty} = 0.0;
    var query_index = 0u;
    loop {{
        if (query_index >= parameters.dimensions.y) {{ break; }}
        increment += weights[physical(parameters.weights, batch, query_index, sequence)] *
            grad_output[physical(parameters.grad_output, batch, query_index, feature)];
        query_index += 1u;
    }}"#,
                    ty = T::TYPE_TOKEN,
                ),
                AttentionSemanticStatus::NonFiniteValueGradient.code(),
                AttentionSemanticStatus::NonFiniteValueGradientArithmetic.code(),
            ),
        };
    format!(
        r#"{prelude}
{bindings}

fn finite(number: {ty}) -> bool {{
    return number == number && abs(number) <= {finite_max};
}}

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = {elements};
    if (id.x >= elements) {{ return; }}
    {coordinates}
    {increment}
    if (!finite(increment)) {{ atomicMin(&status, {arithmetic_failure}u); }}
    let index = physical(parameters.destination, batch, sequence, feature);
    let current = destination[index];
    if (!finite(current)) {{ atomicMin(&status, {destination_failure}u); }}
    if (!finite(current + increment)) {{ atomicMin(&status, {arithmetic_failure}u); }}
}}
"#,
        ty = T::TYPE_TOKEN,
        finite_max = T::FINITE_MAX,
    )
}
