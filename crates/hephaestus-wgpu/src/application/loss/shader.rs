use hephaestus_core::{CrossEntropyStatus, DialectScalar, Wgsl};

pub(super) const FORWARD_PREFLIGHT: u8 = 0;
pub(super) const FORWARD_ROWS: u8 = 1;
pub(super) const FORWARD_MEAN: u8 = 2;
pub(super) const BACKWARD_ROWS: u8 = 3;
pub(super) const BACKWARD_ARITHMETIC: u8 = 4;
pub(super) const BACKWARD_ACCUMULATE: u8 = 5;

/// Scalar spellings for the WGSL cross-entropy shaders: the element type
/// token, the finite bound literal, the `var` annotation WGSL inference
/// needs for f64 (abstract literals default to f32), and the width-selected
/// tolerance/upstream spellings. Naga rejects every f64 bitcast (scalar and
/// vector forms), so the f64 spellings avoid bit tricks entirely: the
/// tolerance is derived arithmetically from the class count and the upstream
/// gradient is re-read from its buffer.
pub(super) trait WgslCrossEntropyScalar: DialectScalar<Wgsl> {
    /// Largest finite literal (`3.402823466e+38` / `1.7976931348623157e+308`).
    const FINITE_MAX: &'static str;
    /// Row-maximum seed literal with sign.
    const NEG_FINITE_MAX: &'static str;
    /// `var`/`let` annotation forcing f64 inference (`""` / `" : f64"`).
    const TYPE_ANNOTATION: &'static str;
    /// Tolerance expression (`bitcast<f32>(z)` / the in-shader gamma formula
    /// from `hephaestus_core::domain::loss::plan`, evaluated in host order so
    /// the IEEE-double result is bit-identical to the plan value).
    const TOLERANCE_LOAD: &'static str;
    /// Upstream stash statements for the backward row preflight (f32 stores
    /// the bits into `status[1]`; f64 emits nothing and re-reads the buffer).
    const UPSTREAM_STORE: &'static str;
    /// Upstream restore expression for the arithmetic preflight (status-word
    /// bitcast / the same direct buffer read the accumulate pass uses).
    const UPSTREAM_LOAD: &'static str;
    /// Extra upstream storage binding for the arithmetic preflight (`None`
    /// for f32; `Some(4)` for f64, with the uniform following at `binding + 1`).
    const ARITHMETIC_UPSTREAM: Option<u32>;
    /// Target-indicator `select` expression. Abstract `0.0`/`1.0` arms
    /// concretize to f32, so f64 spells the arms `f64(...)` explicitly.
    const INDICATOR_SELECT: &'static str;
    /// Backward status words: code + one f32 stash (`2`), or code only (`1`).
    const BACKWARD_STATUS_WORDS: usize;
}

impl WgslCrossEntropyScalar for f32 {
    const FINITE_MAX: &'static str = "3.402823466e+38";
    const NEG_FINITE_MAX: &'static str = "-3.402823466e+38";
    const TYPE_ANNOTATION: &'static str = "";
    const TOLERANCE_LOAD: &'static str = "bitcast<f32>(parameters.dimensions.z)";
    const UPSTREAM_STORE: &'static str = "atomicStore(&status[1], bitcast<u32>(upstream));";
    const UPSTREAM_LOAD: &'static str = "bitcast<f32>(atomicLoad(&status[1]))";
    const ARITHMETIC_UPSTREAM: Option<u32> = None;
    const INDICATOR_SELECT: &'static str = "select(0.0, 1.0, class_index == target_index)";
    const BACKWARD_STATUS_WORDS: usize = 2;
}

impl WgslCrossEntropyScalar for f64 {
    const FINITE_MAX: &'static str = "1.7976931348623157e+308";
    const NEG_FINITE_MAX: &'static str = "-1.7976931348623157e+308";
    const TYPE_ANNOTATION: &'static str = " : f64";
    // `se = eps * (classes - 1)`, `gamma = se / (1 - se)`,
    // `tolerance = gamma + eps * (1 + gamma)`: the same operations in the
    // same order as the host plan, so every correctly-rounded IEEE-double
    // step matches bit-for-bit. The `2.22...e-16` literal is `f64::EPSILON`
    // (pinned by unit test); classes >= 1 is a plan invariant.
    const TOLERANCE_LOAD: &'static str = "((2.220446049250313e-16 * f64(parameters.dimensions.y - 1u)) / (1.0 - (2.220446049250313e-16 * f64(parameters.dimensions.y - 1u)))) + 2.220446049250313e-16 * (1.0 + (((2.220446049250313e-16 * f64(parameters.dimensions.y - 1u)) / (1.0 - (2.220446049250313e-16 * f64(parameters.dimensions.y - 1u))))))";
    const UPSTREAM_STORE: &'static str = "";
    const UPSTREAM_LOAD: &'static str =
        "output_gradient[physical(parameters.output_gradient, 0u, 0u)]";
    const ARITHMETIC_UPSTREAM: Option<u32> = Some(4);
    const INDICATOR_SELECT: &'static str =
        "select(f64(0.0), f64(1.0), class_index == target_index)";
    const BACKWARD_STATUS_WORDS: usize = 1;
}

pub(super) fn shader<T: WgslCrossEntropyScalar>(stage: u8, width: u32) -> String {
    let body = match stage {
        FORWARD_PREFLIGHT => forward_preflight::<T>(),
        FORWARD_ROWS => forward_rows::<T>(),
        FORWARD_MEAN => forward_mean::<T>(),
        BACKWARD_ROWS => backward_rows::<T>(),
        BACKWARD_ARITHMETIC => backward_arithmetic::<T>(),
        BACKWARD_ACCUMULATE => backward_accumulate::<T>(),
        _ => unreachable!("invariant: cross-entropy stage is internal"),
    };
    format!("{}\n{body}", prelude::<T>(width))
}

fn prelude<T: WgslCrossEntropyScalar>(width: u32) -> String {
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

fn finite(value: {ty}) -> bool {{
    return value == value && abs(value) <= {finite_max};
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
        ty = T::TYPE_TOKEN,
        finite_max = T::FINITE_MAX,
        nonfinite_logits = CrossEntropyStatus::NonFiniteLogits.code(),
        target_out_of_range = CrossEntropyStatus::TargetOutOfRange.code(),
        nonfinite_forward = CrossEntropyStatus::NonFiniteForwardArithmetic.code(),
        nonfinite_output_gradient = CrossEntropyStatus::NonFiniteOutputGradient.code(),
        invalid_probabilities = CrossEntropyStatus::InvalidProbabilities.code(),
        nonfinite_destination = CrossEntropyStatus::NonFiniteGradientDestination.code(),
        nonfinite_backward = CrossEntropyStatus::NonFiniteBackwardArithmetic.code(),
    )
}

fn forward_preflight<T: WgslCrossEntropyScalar>() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> logits: array<{ty}>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(3) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let row = id.x;
    if (row >= parameters.dimensions.x) {{ return; }}
    let target_index = targets[physical(parameters.targets, row, 0u)];
    if (target_index >= parameters.dimensions.y) {{
        atomicMin(&status, STATUS_TARGET_OUT_OF_RANGE);
        return;
    }}
    var maximum{annot} = {neg_max};
    var class_index = 0u;
    loop {{
        if (class_index >= parameters.dimensions.y) {{ break; }}
        let value = logits[physical(parameters.logits, row, class_index)];
        if (!finite(value)) {{ atomicMin(&status, STATUS_NON_FINITE_LOGITS); }}
        maximum = max(maximum, value);
        class_index += 1u;
    }}
    var denominator{annot} = 0.0;
    class_index = 0u;
    loop {{
        if (class_index >= parameters.dimensions.y) {{ break; }}
        denominator += exp(logits[physical(parameters.logits, row, class_index)] - maximum);
        class_index += 1u;
    }}
    let target_logit = logits[physical(parameters.logits, row, target_index)];
    let row_loss = log(denominator) + (maximum - target_logit);
    if (!finite(denominator) || denominator <= 0.0 || !finite(row_loss)) {{
        atomicMin(&status, STATUS_NON_FINITE_FORWARD);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        annot = T::TYPE_ANNOTATION,
        neg_max = T::NEG_FINITE_MAX,
    )
}

fn forward_rows<T: WgslCrossEntropyScalar>() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> logits: array<{ty}>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> probabilities: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> row_losses: array<{ty}>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let row = id.x;
    if (row >= parameters.dimensions.x) {{ return; }}
    var maximum{annot} = {neg_max};
    var class_index = 0u;
    loop {{
        if (class_index >= parameters.dimensions.y) {{ break; }}
        maximum = max(maximum, logits[physical(parameters.logits, row, class_index)]);
        class_index += 1u;
    }}
    var denominator{annot} = 0.0;
    class_index = 0u;
    loop {{
        if (class_index >= parameters.dimensions.y) {{ break; }}
        denominator += exp(logits[physical(parameters.logits, row, class_index)] - maximum);
        class_index += 1u;
    }}
    class_index = 0u;
    loop {{
        if (class_index >= parameters.dimensions.y) {{ break; }}
        probabilities[physical(parameters.probabilities, row, class_index)] =
            exp(logits[physical(parameters.logits, row, class_index)] - maximum) / denominator;
        class_index += 1u;
    }}
    let target_index = targets[physical(parameters.targets, row, 0u)];
    row_losses[row] = log(denominator) +
        (maximum - logits[physical(parameters.logits, row, target_index)]);
}}
"#,
        ty = T::TYPE_TOKEN,
        annot = T::TYPE_ANNOTATION,
        neg_max = T::NEG_FINITE_MAX,
    )
}

fn forward_mean<T: WgslCrossEntropyScalar>() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> row_losses: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> loss: array<{ty}>;
@group(0) @binding(2) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(1)
fn main() {{
    var mean{annot} = 0.0;
    var row = 0u;
    loop {{
        if (row >= parameters.dimensions.x) {{ break; }}
        mean += (row_losses[row] - mean) / {ty}(row + 1u);
        row += 1u;
    }}
    loss[physical(parameters.loss, 0u, 0u)] = mean;
}}
"#,
        ty = T::TYPE_TOKEN,
        annot = T::TYPE_ANNOTATION,
    )
}

fn backward_rows<T: WgslCrossEntropyScalar>() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> output_gradient: array<{ty}>;
@group(0) @binding(1) var<storage, read> probabilities: array<{ty}>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let row = id.x;
    if (row >= parameters.dimensions.x) {{ return; }}
    let upstream = output_gradient[physical(parameters.output_gradient, 0u, 0u)];
    if (row == 0u) {{ {upstream_store} }}
    if (!finite(upstream)) {{ atomicMin(&status[0], STATUS_NON_FINITE_OUTPUT_GRADIENT); }}
    if (targets[physical(parameters.targets, row, 0u)] >= parameters.dimensions.y) {{
        atomicMin(&status[0], STATUS_TARGET_OUT_OF_RANGE);
    }}
    var sum{annot} = 0.0;
    var class_index = 0u;
    loop {{
        if (class_index >= parameters.dimensions.y) {{ break; }}
        let probability = probabilities[physical(parameters.probabilities, row, class_index)];
        if (!finite(probability) || probability < 0.0 || probability > 1.0) {{
            atomicMin(&status[0], STATUS_INVALID_PROBABILITIES);
        }}
        sum += probability;
        class_index += 1u;
    }}
    let tolerance = {tolerance};
    if (!finite(sum) || abs(sum - 1.0) > tolerance) {{
        atomicMin(&status[0], STATUS_INVALID_PROBABILITIES);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        annot = T::TYPE_ANNOTATION,
        tolerance = T::TOLERANCE_LOAD,
        upstream_store = T::UPSTREAM_STORE,
    )
}

fn backward_arithmetic<T: WgslCrossEntropyScalar>() -> String {
    let (upstream_decl, uniform) = match T::ARITHMETIC_UPSTREAM {
        Some(binding) => (
            format!(
                "@group(0) @binding({binding}) var<storage, read> output_gradient: array<{ty}>;\n",
                ty = T::TYPE_TOKEN,
            ),
            binding + 1,
        ),
        None => (String::new(), 4),
    };
    format!(
        r#"
@group(0) @binding(0) var<storage, read> probabilities: array<{ty}>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> destination: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
{upstream_decl}@group(0) @binding({uniform}) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= elements) {{ return; }}
    let row = id.x / parameters.dimensions.y;
    let class_index = id.x % parameters.dimensions.y;
    let upstream = {upstream_load};
    let scaled = upstream / {ty}(parameters.dimensions.x);
    let probability = probabilities[physical(parameters.probabilities, row, class_index)];
    let target_index = targets[physical(parameters.targets, row, 0u)];
    let indicator{annot} = {indicator};
    let increment = scaled * (probability - indicator);
    let current = destination[physical(parameters.logit_gradient, row, class_index)];
    if (!finite(current)) {{ atomicMin(&status[0], STATUS_NON_FINITE_DESTINATION); }}
    if (!finite(increment) || !finite(current + increment)) {{
        atomicMin(&status[0], STATUS_NON_FINITE_BACKWARD);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        annot = T::TYPE_ANNOTATION,
        upstream_load = T::UPSTREAM_LOAD,
        upstream_decl = upstream_decl,
        uniform = uniform,
        indicator = T::INDICATOR_SELECT,
    )
}

fn backward_accumulate<T: WgslCrossEntropyScalar>() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> output_gradient: array<{ty}>;
@group(0) @binding(1) var<storage, read> probabilities: array<{ty}>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(4) var<uniform> parameters: CrossEntropyMeta;

@compute @workgroup_size(WORKGROUP_WIDTH)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = parameters.dimensions.x * parameters.dimensions.y;
    if (id.x >= elements) {{ return; }}
    let row = id.x / parameters.dimensions.y;
    let class_index = id.x % parameters.dimensions.y;
    let target_index = targets[physical(parameters.targets, row, 0u)];
    let indicator{annot} = {indicator};
    let upstream = output_gradient[physical(parameters.output_gradient, 0u, 0u)];
    let index = physical(parameters.logit_gradient, row, class_index);
    destination[index] += upstream *
        (probabilities[physical(parameters.probabilities, row, class_index)] - indicator) /
        {ty}(parameters.dimensions.x);
}}
"#,
        ty = T::TYPE_TOKEN,
        annot = T::TYPE_ANNOTATION,
        indicator = T::INDICATOR_SELECT,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_sources_keep_the_stable_spellings() {
        let preflight = shader::<f32>(FORWARD_PREFLIGHT, 256);
        assert!(preflight.contains("logits: array<f32>;"));
        assert!(preflight.contains("var maximum = -3.402823466e+38;"));
        assert!(preflight.contains("var denominator = 0.0;"));
        assert!(preflight.contains("fn finite(value: f32) -> bool"));
        let mean = shader::<f32>(FORWARD_MEAN, 256);
        assert!(mean.contains("mean += (row_losses[row] - mean) / f32(row + 1u);"));
        let rows = shader::<f32>(BACKWARD_ROWS, 256);
        assert!(rows.contains("let tolerance = bitcast<f32>(parameters.dimensions.z);"));
        assert!(rows.contains("atomicStore(&status[1], bitcast<u32>(upstream));"));
        let arithmetic = shader::<f32>(BACKWARD_ARITHMETIC, 256);
        assert!(arithmetic.contains("let upstream = bitcast<f32>(atomicLoad(&status[1]));"));
        assert!(
            arithmetic.contains("let indicator = select(0.0, 1.0, class_index == target_index);")
        );
    }

    #[test]
    fn f64_sources_type_storage_f64_and_annotate_inference() {
        for stage in [
            FORWARD_PREFLIGHT,
            FORWARD_ROWS,
            FORWARD_MEAN,
            BACKWARD_ROWS,
            BACKWARD_ARITHMETIC,
            BACKWARD_ACCUMULATE,
        ] {
            let source = shader::<f64>(stage, 256);
            assert!(
                !source.contains("array<f32>"),
                "stage {stage} still declares f32 storage"
            );
            assert!(
                !source.contains("f32("),
                "stage {stage} still casts through f32"
            );
        }
        let preflight = shader::<f64>(FORWARD_PREFLIGHT, 256);
        assert!(preflight.contains("logits: array<f64>;"));
        assert!(preflight.contains("var maximum : f64 = -1.7976931348623157e+308;"));
        assert!(preflight.contains("var denominator : f64 = 0.0;"));
        assert!(preflight.contains("fn finite(value: f64) -> bool"));
        assert!(preflight.contains("abs(value) <= 1.7976931348623157e+308"));
        let mean = shader::<f64>(FORWARD_MEAN, 256);
        assert!(mean.contains("var mean : f64 = 0.0;"));
        assert!(mean.contains("mean += (row_losses[row] - mean) / f64(row + 1u);"));
    }

    #[test]
    fn f64_backward_derives_tolerance_and_upstream_without_bitcasts() {
        // The WGSL epsilon literal must stay the exact `f64::EPSILON`
        // spelling, or the in-shader gamma formula drifts from the plan.
        assert_eq!(format!("{:e}", f64::EPSILON), "2.220446049250313e-16");
        for stage in [
            FORWARD_PREFLIGHT,
            FORWARD_ROWS,
            FORWARD_MEAN,
            BACKWARD_ROWS,
            BACKWARD_ARITHMETIC,
            BACKWARD_ACCUMULATE,
        ] {
            let source = shader::<f64>(stage, 256);
            assert!(
                !source.contains("bitcast"),
                "stage {stage} uses a naga-rejected f64 bitcast"
            );
            assert!(
                !source.contains("u64"),
                "stage {stage} uses u64 outside the f64 protocol"
            );
        }
        let rows = shader::<f64>(BACKWARD_ROWS, 256);
        assert!(rows.contains("f64(parameters.dimensions.y - 1u)"));
        assert!(rows.contains("2.220446049250313e-16"));
        assert!(!rows.contains("atomicStore(&status[1]"));
        let arithmetic = shader::<f64>(BACKWARD_ARITHMETIC, 256);
        assert!(
            arithmetic
                .contains("@group(0) @binding(4) var<storage, read> output_gradient: array<f64>;")
        );
        assert!(arithmetic.contains("@group(0) @binding(5) var<uniform> parameters:"));
        assert!(arithmetic.contains(
            "let upstream = output_gradient[physical(parameters.output_gradient, 0u, 0u)];"
        ));
        assert!(arithmetic.contains(
            "let indicator : f64 = select(f64(0.0), f64(1.0), class_index == target_index);"
        ));
        let accumulate = shader::<f64>(BACKWARD_ACCUMULATE, 256);
        assert!(accumulate.contains(
            "let indicator : f64 = select(f64(0.0), f64(1.0), class_index == target_index);"
        ));
        assert!(accumulate.contains("/\n        f64(parameters.dimensions.x);"));
    }
}
