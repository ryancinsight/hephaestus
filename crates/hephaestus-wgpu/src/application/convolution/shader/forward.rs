use super::common::prelude;
use super::{BiasMode, ConvolutionDirection};

pub(in super::super) fn forward_shader(
    scalar: &str,
    direction: ConvolutionDirection,
    bias_mode: BiasMode,
    width: u32,
) -> String {
    let (bias_binding, output_binding, metadata_binding, bias_addition) = match bias_mode {
        BiasMode::Absent => (String::new(), 2, 3, ""),
        BiasMode::Present => (
            format!("@group(0) @binding(2) var<storage, read> bias: array<{scalar}>;\n"),
            3,
            4,
            "value += bias[physical(parameters.destination, bias_coordinates)];",
        ),
    };
    let projection = match direction {
        ConvolutionDirection::Regular => regular_projection(),
        ConvolutionDirection::Transposed => transposed_projection(),
    };
    let weight_channels = match direction {
        ConvolutionDirection::Regular => {
            "weight_coordinates[0] = output_coordinates[1];\n            weight_coordinates[1] = input_channel;"
        }
        ConvolutionDirection::Transposed => {
            "weight_coordinates[0] = input_channel;\n            weight_coordinates[1] = output_coordinates[1];"
        }
    };

    format!(
        r#"{prelude}
@group(0) @binding(0) var<storage, read> input: array<{scalar}>;
@group(0) @binding(1) var<storage, read> weight: array<{scalar}>;
{bias_binding}@group(0) @binding({output_binding}) var<storage, read_write> output: array<{scalar}>;
@group(0) @binding({metadata_binding}) var<uniform> parameters: ConvolutionMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let output_elements =
        layout_extent(parameters.output, 0u) *
        layout_extent(parameters.output, 1u) *
        spatial_elements(parameters.output);
    if (id.x >= output_elements) {{
        return;
    }}

    let output_coordinates = decode_layout(id.x, parameters.output);
    let spatial_rank = parameters.spatial_rank_and_flags.x;
    let kernel_elements = spatial_elements(parameters.weight);
    let input_channels = layout_extent(parameters.input, 1u);
    var value = zero();

    var input_channel = 0u;
    loop {{
        if (input_channel >= input_channels) {{
            break;
        }}
        var kernel_linear = 0u;
        loop {{
            if (kernel_linear >= kernel_elements) {{
                break;
            }}
            let kernel_coordinates = decode_spatial(kernel_linear, parameters.weight);
            var input_spatial: array<u32, 3>;
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
                let input_coordinates = coordinates_with_spatial(
                    output_coordinates[0],
                    input_channel,
                    input_spatial,
                    parameters.input.offset_and_rank.y,
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
                value += input[physical(parameters.input, input_coordinates)] *
                    weight[physical(parameters.weight, weight_coordinates)];
            }}
            kernel_linear += 1u;
        }}
        input_channel += 1u;
    }}

    var bias_coordinates: array<u32, 5>;
    bias_coordinates[0] = output_coordinates[1];
    {bias_addition}
    output[physical(parameters.output, output_coordinates)] = value;
}}
"#,
        prelude = prelude(scalar),
    )
}

fn regular_projection() -> &'static str {
    r#"
                let projected =
                    i32(output_coordinates[axis + 2u]) *
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

fn transposed_projection() -> &'static str {
    r#"
                let numerator =
                    i32(output_coordinates[axis + 2u]) +
                    i32(parameter(parameters.padding_low, parameters.padding_high, axis)) -
                    i32(kernel_coordinates[axis]) *
                        i32(parameter(parameters.dilation_low, parameters.dilation_high, axis));
                let axis_stride =
                    i32(parameter(parameters.stride_low, parameters.stride_high, axis));
                if (numerator < 0 || numerator % axis_stride != 0) {
                    valid = false;
                } else {
                    let projected = numerator / axis_stride;
                    if (projected >= i32(layout_extent(parameters.input, axis + 2u))) {
                        valid = false;
                    } else {
                        input_spatial[axis] = u32(projected);
                    }
                }
"#
}
