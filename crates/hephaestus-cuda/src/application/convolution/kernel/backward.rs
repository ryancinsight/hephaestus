use super::prelude::prelude;
use crate::application::convolution::routing::{ConvolutionDirection, GradientTarget};

pub(in super::super) fn backward_source(
    scalar: &str,
    entry: &str,
    direction: ConvolutionDirection,
    target: GradientTarget,
    spatial_rank: usize,
) -> String {
    match target {
        GradientTarget::Input => input_gradient_source(scalar, entry, direction, spatial_rank),
        GradientTarget::Weight => weight_gradient_source(scalar, entry, direction, spatial_rank),
        GradientTarget::Bias => bias_gradient_source(scalar, entry),
    }
}

fn input_gradient_source(
    scalar: &str,
    entry: &str,
    direction: ConvolutionDirection,
    spatial_rank: usize,
) -> String {
    let projection = match direction {
        ConvolutionDirection::Regular => regular_input_projection(),
        ConvolutionDirection::Transposed => transposed_input_projection(),
    };
    let weight_channels = match direction {
        ConvolutionDirection::Regular => {
            "weight_coordinates[0] = output_channel;\n\
             weight_coordinates[1] = input_coordinates[1];"
        }
        ConvolutionDirection::Transposed => {
            "weight_coordinates[0] = input_coordinates[1];\n\
             weight_coordinates[1] = output_channel;"
        }
    };

    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* weight,
    {scalar}* grad_input,
    ConvolutionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int target_elements =
        parameters.destination.shape[0] *
        parameters.destination.shape[1] *
        spatial_elements(parameters.destination);
    if (linear >= target_elements) {{
        return;
    }}

    int input_coordinates[5];
    decode_layout(linear, parameters.destination, input_coordinates);
    const int kernel_elements = spatial_elements(parameters.weight);
    const int output_channels = parameters.output.shape[1];
    {scalar} value = ({scalar})0;

    for (int output_channel = 0; output_channel < output_channels; ++output_channel) {{
        for (int kernel_linear = 0; kernel_linear < kernel_elements; ++kernel_linear) {{
            int kernel_coordinates[3];
            decode_spatial(kernel_linear, parameters.weight, kernel_coordinates);
            int output_spatial[3] = {{0, 0, 0}};
            bool valid = true;
            for (int axis = 0; axis < {spatial_rank}; ++axis) {{
                {projection}
            }}
            if (!valid) {{
                continue;
            }}

            int output_coordinates[5];
            coordinates_with_spatial(
                input_coordinates[0],
                output_channel,
                output_spatial,
                parameters.output.rank,
                output_coordinates
            );
            int weight_coordinates[5] = {{0, 0, 0, 0, 0}};
            {weight_channels}
            for (int axis = 0; axis < {spatial_rank}; ++axis) {{
                weight_coordinates[axis + 2] = kernel_coordinates[axis];
            }}
            value += grad_output[physical(parameters.output, output_coordinates)] *
                weight[physical(parameters.weight, weight_coordinates)];
        }}
    }}

    grad_input[physical(parameters.destination, input_coordinates)] += value;
}}
"#,
        prelude = prelude(),
    )
}

fn weight_gradient_source(
    scalar: &str,
    entry: &str,
    direction: ConvolutionDirection,
    spatial_rank: usize,
) -> String {
    let (input_channel, output_channel, source_layout, projection) = match direction {
        ConvolutionDirection::Regular => (
            "weight_coordinates[1]",
            "weight_coordinates[0]",
            "parameters.output",
            regular_weight_projection(),
        ),
        ConvolutionDirection::Transposed => (
            "weight_coordinates[0]",
            "weight_coordinates[1]",
            "parameters.input",
            transposed_weight_projection(),
        ),
    };

    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    const {scalar}* input,
    {scalar}* grad_weight,
    ConvolutionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int target_elements =
        parameters.destination.shape[0] *
        parameters.destination.shape[1] *
        spatial_elements(parameters.destination);
    if (linear >= target_elements) {{
        return;
    }}

    int weight_coordinates[5];
    decode_layout(linear, parameters.destination, weight_coordinates);
    int kernel_coordinates[3] = {{0, 0, 0}};
    for (int axis = 0; axis < {spatial_rank}; ++axis) {{
        kernel_coordinates[axis] = weight_coordinates[axis + 2];
    }}

    const int batch_extent = parameters.input.shape[0];
    const int source_spatial_elements = spatial_elements({source_layout});
    {scalar} value = ({scalar})0;
    for (int batch = 0; batch < batch_extent; ++batch) {{
        for (int source_linear = 0;
             source_linear < source_spatial_elements;
             ++source_linear) {{
            int source_spatial[3];
            decode_spatial(source_linear, {source_layout}, source_spatial);
            int input_spatial[3] = {{0, 0, 0}};
            int output_spatial[3] = {{0, 0, 0}};
            bool valid = true;
            for (int axis = 0; axis < {spatial_rank}; ++axis) {{
                {projection}
            }}
            if (!valid) {{
                continue;
            }}

            int input_coordinates[5];
            coordinates_with_spatial(
                batch,
                {input_channel},
                input_spatial,
                parameters.input.rank,
                input_coordinates
            );
            int output_coordinates[5];
            coordinates_with_spatial(
                batch,
                {output_channel},
                output_spatial,
                parameters.output.rank,
                output_coordinates
            );
            value += input[physical(parameters.input, input_coordinates)] *
                grad_output[physical(parameters.output, output_coordinates)];
        }}
    }}

    grad_weight[physical(parameters.destination, weight_coordinates)] += value;
}}
"#,
        prelude = prelude(),
    )
}

fn bias_gradient_source(scalar: &str, entry: &str) -> String {
    format!(
        r#"{prelude}
extern "C" __global__ void {entry}(
    const {scalar}* grad_output,
    {scalar}* grad_bias,
    ConvolutionMeta parameters
) {{
    const int output_channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_channel >= parameters.output.shape[1]) {{
        return;
    }}

    const int batch_extent = parameters.output.shape[0];
    const int output_spatial_elements = spatial_elements(parameters.output);
    {scalar} value = ({scalar})0;
    for (int batch = 0; batch < batch_extent; ++batch) {{
        for (int output_linear = 0;
             output_linear < output_spatial_elements;
             ++output_linear) {{
            int output_spatial[3];
            decode_spatial(output_linear, parameters.output, output_spatial);
            int output_coordinates[5];
            coordinates_with_spatial(
                batch,
                output_channel,
                output_spatial,
                parameters.output.rank,
                output_coordinates
            );
            value += grad_output[physical(parameters.output, output_coordinates)];
        }}
    }}

    int target_coordinates[5] = {{output_channel, 0, 0, 0, 0}};
    grad_bias[physical(parameters.destination, target_coordinates)] += value;
}}
"#,
        prelude = prelude(),
    )
}

fn regular_input_projection() -> &'static str {
    r#"
                const int numerator =
                    input_coordinates[axis + 2] + parameters.padding[axis] -
                    kernel_coordinates[axis] * parameters.dilation[axis];
                const int axis_stride = parameters.stride[axis];
                if (numerator < 0 || numerator % axis_stride != 0) {
                    valid = false;
                    break;
                }
                const int projected = numerator / axis_stride;
                if (projected >= parameters.output.shape[axis + 2]) {
                    valid = false;
                    break;
                }
                output_spatial[axis] = projected;
"#
}

fn transposed_input_projection() -> &'static str {
    r#"
                const int projected =
                    input_coordinates[axis + 2] * parameters.stride[axis] +
                    kernel_coordinates[axis] * parameters.dilation[axis] -
                    parameters.padding[axis];
                if (projected < 0 || projected >= parameters.output.shape[axis + 2]) {
                    valid = false;
                    break;
                }
                output_spatial[axis] = projected;
"#
}

fn regular_weight_projection() -> &'static str {
    r#"
                output_spatial[axis] = source_spatial[axis];
                const int projected =
                    source_spatial[axis] * parameters.stride[axis] +
                    kernel_coordinates[axis] * parameters.dilation[axis] -
                    parameters.padding[axis];
                if (projected < 0 || projected >= parameters.input.shape[axis + 2]) {
                    valid = false;
                    break;
                }
                input_spatial[axis] = projected;
"#
}

fn transposed_weight_projection() -> &'static str {
    r#"
                input_spatial[axis] = source_spatial[axis];
                const int projected =
                    source_spatial[axis] * parameters.stride[axis] +
                    kernel_coordinates[axis] * parameters.dilation[axis] -
                    parameters.padding[axis];
                if (projected < 0 || projected >= parameters.output.shape[axis + 2]) {
                    valid = false;
                    break;
                }
                output_spatial[axis] = projected;
"#
}
