use super::ConvolutionDirection;
use super::prelude::prelude;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum GradientTarget {
    Input,
    Weight,
    Bias,
}

pub(in super::super) fn backward_shader(
    scalar: &str,
    direction: ConvolutionDirection,
    target: GradientTarget,
    width: u32,
) -> String {
    match target {
        GradientTarget::Input => input_gradient_shader(scalar, direction, width),
        GradientTarget::Weight => weight_gradient_shader(scalar, direction, width),
        GradientTarget::Bias => bias_gradient_shader(scalar, width),
    }
}

fn input_gradient_shader(scalar: &str, direction: ConvolutionDirection, width: u32) -> String {
    let projection = match direction {
        ConvolutionDirection::Regular => regular_input_projection(),
        ConvolutionDirection::Transposed => transposed_input_projection(),
    };
    let weight_channels = match direction {
        ConvolutionDirection::Regular => {
            "weight_coordinates[0] = output_channel;\n            weight_coordinates[1] = input_coordinates[1];"
        }
        ConvolutionDirection::Transposed => {
            "weight_coordinates[0] = input_coordinates[1];\n            weight_coordinates[1] = output_channel;"
        }
    };
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> grad_output: array<{scalar}>;
@group(0) @binding(1) var<storage, read> weight: array<{scalar}>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<{scalar}>;
@group(0) @binding(3) var<uniform> parameters: ConvolutionMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let target_elements =
        layout_extent(parameters.destination, 0u) *
        layout_extent(parameters.destination, 1u) *
        spatial_elements(parameters.destination);
    if (id.x >= target_elements) {{
        return;
    }}

    let input_coordinates = decode_layout(id.x, parameters.destination);
    let spatial_rank = parameters.spatial_rank_and_flags.x;
    let kernel_elements = spatial_elements(parameters.weight);
    let output_channels = layout_extent(parameters.output, 1u);
    var value = zero();

    var output_channel = 0u;
    loop {{
        if (output_channel >= output_channels) {{
            break;
        }}
        var kernel_linear = 0u;
        loop {{
            if (kernel_linear >= kernel_elements) {{
                break;
            }}
            let kernel_coordinates = decode_spatial(kernel_linear, parameters.weight);
            var output_spatial: array<u32, 3>;
            var valid = true;
            var axis = 0u;
            loop {{
                if (axis >= spatial_rank) {{
                    break;
                }}
                {projection}
                axis += 1u;
            }}
            if (valid) {{
                let output_coordinates = coordinates_with_spatial(
                    input_coordinates[0],
                    output_channel,
                    output_spatial,
                    parameters.output.offset_and_rank.y,
                );
                var weight_coordinates: array<u32, 5>;
                {weight_channels}
                axis = 0u;
                loop {{
                    if (axis >= spatial_rank) {{
                        break;
                    }}
                    weight_coordinates[axis + 2u] = kernel_coordinates[axis];
                    axis += 1u;
                }}
                value += grad_output[physical(parameters.output, output_coordinates)] *
                    weight[physical(parameters.weight, weight_coordinates)];
            }}
            kernel_linear += 1u;
        }}
        output_channel += 1u;
    }}

    let target_index = physical(parameters.destination, input_coordinates);
    grad_input[target_index] += value;
}}
"#,
        prelude = prelude(scalar),
    )
}

fn weight_gradient_shader(scalar: &str, direction: ConvolutionDirection, width: u32) -> String {
    let (input_channel, output_channel) = match direction {
        ConvolutionDirection::Regular => ("weight_coordinates[1]", "weight_coordinates[0]"),
        ConvolutionDirection::Transposed => ("weight_coordinates[0]", "weight_coordinates[1]"),
    };
    let source_layout = match direction {
        ConvolutionDirection::Regular => "parameters.output",
        ConvolutionDirection::Transposed => "parameters.input",
    };
    let projection = match direction {
        ConvolutionDirection::Regular => regular_weight_projection(),
        ConvolutionDirection::Transposed => transposed_weight_projection(),
    };

    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> grad_output: array<{scalar}>;
@group(0) @binding(1) var<storage, read> input: array<{scalar}>;
@group(0) @binding(2) var<storage, read_write> grad_weight: array<{scalar}>;
@group(0) @binding(3) var<uniform> parameters: ConvolutionMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let target_elements =
        layout_extent(parameters.destination, 0u) *
        layout_extent(parameters.destination, 1u) *
        spatial_elements(parameters.destination);
    if (id.x >= target_elements) {{
        return;
    }}

    let weight_coordinates = decode_layout(id.x, parameters.destination);
    let spatial_rank = parameters.spatial_rank_and_flags.x;
    var kernel_coordinates: array<u32, 3>;
    var axis = 0u;
    loop {{
        if (axis >= spatial_rank) {{
            break;
        }}
        kernel_coordinates[axis] = weight_coordinates[axis + 2u];
        axis += 1u;
    }}

    let batch_extent = layout_extent(parameters.input, 0u);
    let source_spatial_elements = spatial_elements({source_layout});
    var value = zero();
    var batch = 0u;
    loop {{
        if (batch >= batch_extent) {{
            break;
        }}
        var source_linear = 0u;
        loop {{
            if (source_linear >= source_spatial_elements) {{
                break;
            }}
            let source_spatial = decode_spatial(source_linear, {source_layout});
            var input_spatial: array<u32, 3>;
            var output_spatial: array<u32, 3>;
            var valid = true;
            axis = 0u;
            loop {{
                if (axis >= spatial_rank) {{
                    break;
                }}
                {projection}
                axis += 1u;
            }}
            if (valid) {{
                let input_coordinates = coordinates_with_spatial(
                    batch,
                    {input_channel},
                    input_spatial,
                    parameters.input.offset_and_rank.y,
                );
                let output_coordinates = coordinates_with_spatial(
                    batch,
                    {output_channel},
                    output_spatial,
                    parameters.output.offset_and_rank.y,
                );
                value += input[physical(parameters.input, input_coordinates)] *
                    grad_output[physical(parameters.output, output_coordinates)];
            }}
            source_linear += 1u;
        }}
        batch += 1u;
    }}

    let target_index = physical(parameters.destination, weight_coordinates);
    grad_weight[target_index] += value;
}}
"#,
        prelude = prelude(scalar),
    )
}

fn bias_gradient_shader(scalar: &str, width: u32) -> String {
    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> grad_output: array<{scalar}>;
@group(0) @binding(1) var<storage, read_write> grad_bias: array<{scalar}>;
@group(0) @binding(2) var<uniform> parameters: ConvolutionMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let output_channel = id.x;
    if (output_channel >= layout_extent(parameters.output, 1u)) {{
        return;
    }}

    let batch_extent = layout_extent(parameters.output, 0u);
    let output_spatial_elements = spatial_elements(parameters.output);
    var value = zero();
    var batch = 0u;
    loop {{
        if (batch >= batch_extent) {{
            break;
        }}
        var output_linear = 0u;
        loop {{
            if (output_linear >= output_spatial_elements) {{
                break;
            }}
            let output_spatial = decode_spatial(output_linear, parameters.output);
            let output_coordinates = coordinates_with_spatial(
                batch,
                output_channel,
                output_spatial,
                parameters.output.offset_and_rank.y,
            );
            value += grad_output[physical(parameters.output, output_coordinates)];
            output_linear += 1u;
        }}
        batch += 1u;
    }}

    var target_coordinates: array<u32, 5>;
    target_coordinates[0] = output_channel;
    let target_index = physical(parameters.destination, target_coordinates);
    grad_bias[target_index] += value;
}}
"#,
        prelude = prelude(scalar),
    )
}

fn regular_input_projection() -> &'static str {
    r#"
                let numerator =
                    i32(input_coordinates[axis + 2u]) +
                    i32(parameter(parameters.padding_low, parameters.padding_high, axis)) -
                    i32(kernel_coordinates[axis]) *
                        i32(parameter(parameters.dilation_low, parameters.dilation_high, axis));
                let axis_stride =
                    i32(parameter(parameters.stride_low, parameters.stride_high, axis));
                if (numerator < 0 || numerator % axis_stride != 0) {
                    valid = false;
                } else {
                    let projected = numerator / axis_stride;
                    if (projected >= i32(layout_extent(parameters.output, axis + 2u))) {
                        valid = false;
                    } else {
                        output_spatial[axis] = u32(projected);
                    }
                }
"#
}

fn transposed_input_projection() -> &'static str {
    r#"
                let projected =
                    i32(input_coordinates[axis + 2u]) *
                        i32(parameter(parameters.stride_low, parameters.stride_high, axis)) +
                    i32(kernel_coordinates[axis]) *
                        i32(parameter(parameters.dilation_low, parameters.dilation_high, axis)) -
                    i32(parameter(parameters.padding_low, parameters.padding_high, axis));
                if (
                    projected < 0 ||
                    projected >= i32(layout_extent(parameters.output, axis + 2u))
                ) {
                    valid = false;
                } else {
                    output_spatial[axis] = u32(projected);
                }
"#
}

fn regular_weight_projection() -> &'static str {
    r#"
                output_spatial[axis] = source_spatial[axis];
                let projected =
                    i32(source_spatial[axis]) *
                        i32(parameter(parameters.stride_low, parameters.stride_high, axis)) +
                    i32(kernel_coordinates[axis]) *
                        i32(parameter(parameters.dilation_low, parameters.dilation_high, axis)) -
                    i32(parameter(parameters.padding_low, parameters.padding_high, axis));
                if (
                    projected < 0 ||
                    projected >= i32(layout_extent(parameters.input, axis + 2u))
                ) {
                    valid = false;
                } else {
                    input_spatial[axis] = u32(projected);
                }
"#
}

fn transposed_weight_projection() -> &'static str {
    r#"
                input_spatial[axis] = source_spatial[axis];
                let projected =
                    i32(source_spatial[axis]) *
                        i32(parameter(parameters.stride_low, parameters.stride_high, axis)) +
                    i32(kernel_coordinates[axis]) *
                        i32(parameter(parameters.dilation_low, parameters.dilation_high, axis)) -
                    i32(parameter(parameters.padding_low, parameters.padding_high, axis));
                if (
                    projected < 0 ||
                    projected >= i32(layout_extent(parameters.output, axis + 2u))
                ) {
                    valid = false;
                } else {
                    output_spatial[axis] = u32(projected);
                }
"#
}
