use super::common::prelude;
use crate::application::convolution::routing::{BiasMode, ConvolutionDirection};

pub(in super::super) fn forward_source(
    scalar: &str,
    entry: &str,
    direction: ConvolutionDirection,
    bias: BiasMode,
    spatial_rank: usize,
) -> String {
    let (bias_parameter, bias_addition) = match bias {
        BiasMode::Absent => (String::new(), ""),
        BiasMode::Present => (
            format!("const {scalar}* bias, "),
            r#"int bias_coordinates[5] = {output_coordinates[1], 0, 0, 0, 0};
    value += bias[physical(parameters.destination, bias_coordinates)];"#,
        ),
    };
    let projection = match direction {
        ConvolutionDirection::Regular => regular_projection(),
        ConvolutionDirection::Transposed => transposed_projection(),
    };
    let weight_channels = match direction {
        ConvolutionDirection::Regular => {
            "weight_coordinates[0] = output_coordinates[1];\n\
             weight_coordinates[1] = input_channel;"
        }
        ConvolutionDirection::Transposed => {
            "weight_coordinates[0] = input_channel;\n\
             weight_coordinates[1] = output_coordinates[1];"
        }
    };

    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* input,
    const {scalar}* weight,
    {bias_parameter}{scalar}* output,
    ConvolutionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_elements =
        parameters.output.shape[0] *
        parameters.output.shape[1] *
        spatial_elements(parameters.output);
    if (linear >= output_elements) {{
        return;
    }}

    int output_coordinates[5];
    decode_layout(linear, parameters.output, output_coordinates);
    const int kernel_elements = spatial_elements(parameters.weight);
    const int input_channels = parameters.input.shape[1];
    {scalar} value = ({scalar})0;

    for (int input_channel = 0; input_channel < input_channels; ++input_channel) {{
        for (int kernel_linear = 0; kernel_linear < kernel_elements; ++kernel_linear) {{
            int kernel_coordinates[3];
            decode_spatial(kernel_linear, parameters.weight, kernel_coordinates);
            int input_spatial[3] = {{0, 0, 0}};
            bool valid = true;
            for (int axis = 0; axis < {spatial_rank}; ++axis) {{
                {projection}
            }}
            if (!valid) {{
                continue;
            }}

            int input_coordinates[5];
            coordinates_with_spatial(
                output_coordinates[0],
                input_channel,
                input_spatial,
                parameters.input.rank,
                input_coordinates
            );
            int weight_coordinates[5] = {{0, 0, 0, 0, 0}};
            {weight_channels}
            for (int axis = 0; axis < {spatial_rank}; ++axis) {{
                weight_coordinates[axis + 2] = kernel_coordinates[axis];
            }}
            value += input[physical(parameters.input, input_coordinates)] *
                weight[physical(parameters.weight, weight_coordinates)];
        }}
    }}

    {bias_addition}
    output[physical(parameters.output, output_coordinates)] = value;
}}
"#,
        prelude = prelude(),
    )
}

fn regular_projection() -> &'static str {
    r#"
                const int projected =
                    output_coordinates[axis + 2] * parameters.stride[axis] +
                    kernel_coordinates[axis] * parameters.dilation[axis] -
                    parameters.padding[axis];
                if (projected < 0 || projected >= parameters.input.shape[axis + 2]) {
                    valid = false;
                    break;
                }
                input_spatial[axis] = projected;
"#
}

fn transposed_projection() -> &'static str {
    r#"
                const int numerator =
                    output_coordinates[axis + 2] + parameters.padding[axis] -
                    kernel_coordinates[axis] * parameters.dilation[axis];
                const int axis_stride = parameters.stride[axis];
                if (numerator < 0 || numerator % axis_stride != 0) {
                    valid = false;
                    break;
                }
                const int projected = numerator / axis_stride;
                if (projected >= parameters.input.shape[axis + 2]) {
                    valid = false;
                    break;
                }
                input_spatial[axis] = projected;
"#
}
