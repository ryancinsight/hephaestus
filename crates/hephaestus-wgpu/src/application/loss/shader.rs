use hephaestus_core::CrossEntropyStatus;

pub(super) const FORWARD_PREFLIGHT: u8 = 0;
pub(super) const FORWARD_ROWS: u8 = 1;
pub(super) const FORWARD_MEAN: u8 = 2;
pub(super) const BACKWARD_ROWS: u8 = 3;
pub(super) const BACKWARD_ARITHMETIC: u8 = 4;
pub(super) const BACKWARD_ACCUMULATE: u8 = 5;

pub(super) fn shader(stage: u8, width: u32) -> String {
    let body = match stage {
        FORWARD_PREFLIGHT => forward_preflight(),
        FORWARD_ROWS => forward_rows(),
        FORWARD_MEAN => forward_mean(),
        BACKWARD_ROWS => backward_rows(),
        BACKWARD_ARITHMETIC => backward_arithmetic(),
        BACKWARD_ACCUMULATE => backward_accumulate(),
        _ => unreachable!("invariant: cross-entropy stage is internal"),
    };
    format!("{}\n{body}", prelude(width))
}

fn prelude(width: u32) -> String {
    format!(
        r#"
struct LayoutMeta {{
    shape: vec4<u32>,
    address: vec4<i32>,
}}

struct CrossEntropyMeta {{
    logits: LayoutMeta,
    targets: LayoutMeta,
    loss: LayoutMeta,
    probabilities: LayoutMeta,
    output_gradient: LayoutMeta,
    logit_gradient: LayoutMeta,
    dimensions: vec4<u32>,
}}

fn physical(view: LayoutMeta, first: u32, second: u32) -> u32 {{
    return u32(view.address.z + i32(first) * view.address.x + i32(second) * view.address.y);
}}

fn finite(value: f32) -> bool {{
    return value == value && abs(value) <= 3.402823466e+38;
}}

const WORKGROUP_WIDTH: u32 = {width}u;
const STATUS_NON_FINITE_LOGITS: u32 = {nonfinite_logits}u;
const STATUS_TARGET_OUT_OF_RANGE: u32 = {target_out_of_range}u;
const STATUS_NON_FINITE_FORWARD: u32 = {nonfinite_forward}u;
const STATUS_NON_FINITE_OUTPUT_GRADIENT: u32 = {nonfinite_output_gradient}u;
const STATUS_INVALID_PROBABILITIES: u32 = {invalid_probabilities}u;
const STATUS_NON_FINITE_DESTINATION: u32 = {nonfinite_destination}u;
const STATUS_NON_FINITE_BACKWARD: u32 = {nonfinite_backward}u;
"#,
        nonfinite_logits = CrossEntropyStatus::NonFiniteLogits.code(),
        target_out_of_range = CrossEntropyStatus::TargetOutOfRange.code(),
        nonfinite_forward = CrossEntropyStatus::NonFiniteForwardArithmetic.code(),
        nonfinite_output_gradient = CrossEntropyStatus::NonFiniteOutputGradient.code(),
        invalid_probabilities = CrossEntropyStatus::InvalidProbabilities.code(),
        nonfinite_destination = CrossEntropyStatus::NonFiniteGradientDestination.code(),
        nonfinite_backward = CrossEntropyStatus::NonFiniteBackwardArithmetic.code(),
    )
}

fn forward_preflight() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(3) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= parameters.dimensions.x) { return; }
    let target_index = targets[physical(parameters.targets, row, 0u)];
    if (target_index >= parameters.dimensions.y) {
        atomicMin(&status, STATUS_TARGET_OUT_OF_RANGE);
        return;
    }
    var maximum = -3.402823466e+38;
    var class_index = 0u;
    loop {
        if (class_index >= parameters.dimensions.y) { break; }
        let value = logits[physical(parameters.logits, row, class_index)];
        if (!finite(value)) { atomicMin(&status, STATUS_NON_FINITE_LOGITS); }
        maximum = max(maximum, value);
        class_index += 1u;
    }
    var denominator = 0.0;
    class_index = 0u;
    loop {
        if (class_index >= parameters.dimensions.y) { break; }
        denominator += exp(logits[physical(parameters.logits, row, class_index)] - maximum);
        class_index += 1u;
    }
    let target_logit = logits[physical(parameters.logits, row, target_index)];
    let row_loss = log(denominator) + (maximum - target_logit);
    if (!finite(denominator) || denominator <= 0.0 || !finite(row_loss)) {
        atomicMin(&status, STATUS_NON_FINITE_FORWARD);
    }
}
"#
}

fn forward_rows() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> probabilities: array<f32>;
@group(0) @binding(3) var<storage, read_write> row_losses: array<f32>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= parameters.dimensions.x) { return; }
    var maximum = -3.402823466e+38;
    var class_index = 0u;
    loop {
        if (class_index >= parameters.dimensions.y) { break; }
        maximum = max(maximum, logits[physical(parameters.logits, row, class_index)]);
        class_index += 1u;
    }
    var denominator = 0.0;
    class_index = 0u;
    loop {
        if (class_index >= parameters.dimensions.y) { break; }
        denominator += exp(logits[physical(parameters.logits, row, class_index)] - maximum);
        class_index += 1u;
    }
    class_index = 0u;
    loop {
        if (class_index >= parameters.dimensions.y) { break; }
        probabilities[physical(parameters.probabilities, row, class_index)] =
            exp(logits[physical(parameters.logits, row, class_index)] - maximum) / denominator;
        class_index += 1u;
    }
    let target_index = targets[physical(parameters.targets, row, 0u)];
    row_losses[row] = log(denominator) +
        (maximum - logits[physical(parameters.logits, row, target_index)]);
}
"#
}

fn forward_mean() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> row_losses: array<f32>;
@group(0) @binding(1) var<storage, read_write> loss: array<f32>;
@group(0) @binding(2) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(1)
fn main() {
    var mean = 0.0;
    var row = 0u;
    loop {
        if (row >= parameters.dimensions.x) { break; }
        mean += (row_losses[row] - mean) / f32(row + 1u);
        row += 1u;
    }
    loss[physical(parameters.loss, 0u, 0u)] = mean;
}
"#
}

fn backward_rows() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> output_gradient: array<f32>;
@group(0) @binding(1) var<storage, read> probabilities: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= parameters.dimensions.x) { return; }
    let upstream = output_gradient[physical(parameters.output_gradient, 0u, 0u)];
    if (row == 0u) { atomicStore(&status[1], bitcast<u32>(upstream)); }
    if (!finite(upstream)) { atomicMin(&status[0], STATUS_NON_FINITE_OUTPUT_GRADIENT); }
    if (targets[physical(parameters.targets, row, 0u)] >= parameters.dimensions.y) {
        atomicMin(&status[0], STATUS_TARGET_OUT_OF_RANGE);
    }
    var sum = 0.0;
    var class_index = 0u;
    loop {
        if (class_index >= parameters.dimensions.y) { break; }
        let probability = probabilities[physical(parameters.probabilities, row, class_index)];
        if (!finite(probability) || probability < 0.0 || probability > 1.0) {
            atomicMin(&status[0], STATUS_INVALID_PROBABILITIES);
        }
        sum += probability;
        class_index += 1u;
    }
    let tolerance = bitcast<f32>(parameters.dimensions.z);
    if (!finite(sum) || abs(sum - 1.0) > tolerance) {
        atomicMin(&status[0], STATUS_INVALID_PROBABILITIES);
    }
}
"#
}

fn backward_arithmetic() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> probabilities: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> destination: array<f32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let elements = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= elements) { return; }
    let row = id.x / parameters.dimensions.y;
    let class_index = id.x % parameters.dimensions.y;
    let upstream = bitcast<f32>(atomicLoad(&status[1]));
    let scaled = upstream / f32(parameters.dimensions.x);
    let probability = probabilities[physical(parameters.probabilities, row, class_index)];
    let target_index = targets[physical(parameters.targets, row, 0u)];
    let indicator = select(0.0, 1.0, class_index == target_index);
    let increment = scaled * (probability - indicator);
    let current = destination[physical(parameters.logit_gradient, row, class_index)];
    if (!finite(current)) { atomicMin(&status[0], STATUS_NON_FINITE_DESTINATION); }
    if (!finite(increment) || !finite(current + increment)) {
        atomicMin(&status[0], STATUS_NON_FINITE_BACKWARD);
    }
}
"#
}

fn backward_accumulate() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> output_gradient: array<f32>;
@group(0) @binding(1) var<storage, read> probabilities: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> destination: array<f32>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let elements = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= elements) { return; }
    let row = id.x / parameters.dimensions.y;
    let class_index = id.x % parameters.dimensions.y;
    let target_index = targets[physical(parameters.targets, row, 0u)];
    let indicator = select(0.0, 1.0, class_index == target_index);
    let upstream = output_gradient[physical(parameters.output_gradient, 0u, 0u)];
    let index = physical(parameters.logit_gradient, row, class_index);
    destination[index] += upstream *
        (probabilities[physical(parameters.probabilities, row, class_index)] - indicator) /
        f32(parameters.dimensions.x);
}
"#
}
