use super::WindowOperation;
use crate::domain::dialect::{CudaC, DialectScalar, HipC, KernelDialect};
use crate::domain::launch::BlockWidth;

/// A kernel dialect that can express the generic spatial-window operations.
pub trait WindowDialect: KernelDialect {
    /// Emit one pooling or sliding-window kernel specialized for `T`.
    fn window_source<T: DialectScalar<Self>>(
        operation: WindowOperation,
        width: BlockWidth,
    ) -> String;
}

/// Entry point for a generated spatial-window kernel.
pub const WINDOW_ENTRY: &str = "window_kernel";

/// Generate the C-family pooling and sliding-window kernels.
#[must_use]
pub fn c_family_window_source<L, T>(operation: WindowOperation, width: BlockWidth) -> String
where
    L: KernelDialect,
    T: DialectScalar<L>,
{
    let scalar = T::TYPE_TOKEN;
    let body = match operation {
        WindowOperation::PoolingForwardMaximum => pooling_forward_maximum(scalar),
        WindowOperation::PoolingForwardAverage => pooling_forward_average(scalar),
        WindowOperation::PoolingBackwardMaximum => pooling_backward_maximum(scalar),
        WindowOperation::PoolingBackwardAverage => pooling_backward_average(scalar),
        WindowOperation::Unfold => unfold(scalar),
        WindowOperation::Fold => fold(scalar),
    };
    let arguments = match operation {
        WindowOperation::PoolingForwardMaximum
        | WindowOperation::PoolingForwardAverage
        | WindowOperation::PoolingBackwardAverage
        | WindowOperation::Unfold
        | WindowOperation::Fold => {
            format!("const {scalar}* first, {scalar}* output")
        }
        WindowOperation::PoolingBackwardMaximum => {
            format!("const {scalar}* first, const {scalar}* second, {scalar}* output")
        }
    };
    format!(
        r#"
struct WindowLayoutMeta {{
    unsigned int shape[5];
    int strides[5];
    unsigned int offset;
    unsigned int rank;
}};

struct WindowMeta {{
    WindowLayoutMeta source;
    WindowLayoutMeta target;
    WindowLayoutMeta destination;
    unsigned int kernel[3];
    unsigned int stride[3];
    unsigned int padding[3];
    unsigned int dilation[3];
    unsigned int output_spatial[3];
    unsigned int geometry[4];
}};

__device__ int physical(WindowLayoutMeta layout, int coordinates[5]) {{
    int position = (int)layout.offset;
    for (unsigned int axis = 0u; axis < layout.rank; ++axis) {{
        position += coordinates[axis] * layout.strides[axis];
    }}
    return position;
}}

__device__ int logical_elements(WindowLayoutMeta layout) {{
    int elements = 1;
    for (unsigned int axis = 0u; axis < layout.rank; ++axis) {{
        elements *= (int)layout.shape[axis];
    }}
    return elements;
}}

__device__ void decode_layout(int linear, WindowLayoutMeta layout, int coordinates[5]) {{
    for (unsigned int axis = 0u; axis < 5u; ++axis) {{
        coordinates[axis] = 0;
    }}
    for (int axis = (int)layout.rank - 1; axis >= 0; --axis) {{
        int extent = (int)layout.shape[axis];
        coordinates[axis] = linear % extent;
        linear /= extent;
    }}
}}

__device__ void decode_spatial(
    int linear,
    unsigned int extents[3],
    unsigned int rank,
    int coordinates[3]
) {{
    for (unsigned int axis = 0u; axis < 3u; ++axis) {{
        coordinates[axis] = 0;
    }}
    for (int axis = (int)rank - 1; axis >= 0; --axis) {{
        int extent = (int)extents[axis];
        coordinates[axis] = linear % extent;
        linear /= extent;
    }}
}}

__device__ void with_spatial(
    int batch,
    int channel,
    int spatial[3],
    unsigned int rank,
    int coordinates[5]
) {{
    for (unsigned int axis = 0u; axis < 5u; ++axis) {{
        coordinates[axis] = 0;
    }}
    coordinates[0] = batch;
    coordinates[1] = channel;
    for (unsigned int axis = 0u; axis < rank; ++axis) {{
        coordinates[axis + 2u] = spatial[axis];
    }}
}}

__device__ int spatial_linear(int spatial[3], unsigned int extents[3], unsigned int rank) {{
    int linear = 0;
    for (unsigned int axis = 0u; axis < rank; ++axis) {{
        linear = linear * (int)extents[axis] + spatial[axis];
    }}
    return linear;
}}

__device__ bool project_forward(
    WindowMeta parameters,
    int output_spatial[3],
    int kernel_spatial[3],
    int input_spatial[3]
) {{
    bool valid = true;
    for (unsigned int axis = 0u; axis < parameters.geometry[0]; ++axis) {{
        int projected = output_spatial[axis] * (int)parameters.stride[axis]
            + kernel_spatial[axis] * (int)parameters.dilation[axis]
            - (int)parameters.padding[axis];
        if (projected < 0 || projected >= (int)parameters.source.shape[axis + 2u]) {{
            valid = false;
        }} else {{
            input_spatial[axis] = projected;
        }}
    }}
    return valid;
}}

__device__ bool project_inverse(
    WindowMeta parameters,
    int input_spatial[3],
    int kernel_spatial[3],
    int output_spatial[3]
) {{
    bool valid = true;
    for (unsigned int axis = 0u; axis < parameters.geometry[0]; ++axis) {{
        int numerator = input_spatial[axis] + (int)parameters.padding[axis]
            - kernel_spatial[axis] * (int)parameters.dilation[axis];
        int axis_stride = (int)parameters.stride[axis];
        if (numerator < 0 || numerator % axis_stride != 0) {{
            valid = false;
        }} else {{
            int projected = numerator / axis_stride;
            if (projected >= (int)parameters.output_spatial[axis]) {{
                valid = false;
            }} else {{
                output_spatial[axis] = projected;
            }}
        }}
    }}
    return valid;
}}

extern "C" __global__ void {entry}(
    {arguments},
    WindowMeta parameters
) {{
    // The host launch uses this same width for occupancy planning: {width}.
    int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    {body}
}}
"#,
        entry = WINDOW_ENTRY,
        arguments = arguments,
        width = width.get(),
        body = body,
    )
}

fn pooling_forward_maximum(scalar: &str) -> String {
    format!(
        r#"
    int elements = logical_elements(parameters.target);
    if (linear >= elements) {{ return; }}
    int output_coordinates[5];
    decode_layout(linear, parameters.target, output_coordinates);
    int output_spatial[3] = {{ output_coordinates[2], output_coordinates[3], output_coordinates[4] }};
    {scalar} value = ({scalar})0;
    bool found = false;
    for (unsigned int kernel_linear = 0u; kernel_linear < parameters.geometry[1]; ++kernel_linear) {{
        int kernel_spatial[3];
        decode_spatial(kernel_linear, parameters.kernel, parameters.geometry[0], kernel_spatial);
        int input_spatial[3];
        if (!project_forward(parameters, output_spatial, kernel_spatial, input_spatial)) {{ continue; }}
        int input_coordinates[5];
        with_spatial(output_coordinates[0], output_coordinates[1], input_spatial,
            parameters.source.rank - 2u, input_coordinates);
        {scalar} candidate = first[physical(parameters.source, input_coordinates)];
        if (!found || candidate > value) {{ value = candidate; found = true; }}
    }}
    output[physical(parameters.target, output_coordinates)] = value;
"#,
        scalar = scalar
    )
}

fn pooling_forward_average(scalar: &str) -> String {
    format!(
        r#"
    int elements = logical_elements(parameters.target);
    if (linear >= elements) {{ return; }}
    int output_coordinates[5];
    decode_layout(linear, parameters.target, output_coordinates);
    int output_spatial[3] = {{ output_coordinates[2], output_coordinates[3], output_coordinates[4] }};
    {scalar} value = ({scalar})0;
    unsigned int valid_count = 0u;
    for (unsigned int kernel_linear = 0u; kernel_linear < parameters.geometry[1]; ++kernel_linear) {{
        int kernel_spatial[3];
        decode_spatial(kernel_linear, parameters.kernel, parameters.geometry[0], kernel_spatial);
        int input_spatial[3];
        if (!project_forward(parameters, output_spatial, kernel_spatial, input_spatial)) {{ continue; }}
        int input_coordinates[5];
        with_spatial(output_coordinates[0], output_coordinates[1], input_spatial,
            parameters.source.rank - 2u, input_coordinates);
        value += first[physical(parameters.source, input_coordinates)];
        valid_count += 1u;
    }}
    if (valid_count > 0u) {{ value /= ({scalar})valid_count; }}
    output[physical(parameters.target, output_coordinates)] = value;
"#,
        scalar = scalar
    )
}

fn pooling_backward_average(scalar: &str) -> String {
    format!(
        r#"
    int elements = logical_elements(parameters.destination);
    if (linear >= elements) {{ return; }}
    int destination_coordinates[5];
    decode_layout(linear, parameters.destination, destination_coordinates);
    int input_spatial[3] = {{ destination_coordinates[2], destination_coordinates[3], destination_coordinates[4] }};
    for (unsigned int kernel_linear = 0u; kernel_linear < parameters.geometry[1]; ++kernel_linear) {{
        int kernel_spatial[3];
        decode_spatial(kernel_linear, parameters.kernel, parameters.geometry[0], kernel_spatial);
        int output_spatial[3];
        if (!project_inverse(parameters, input_spatial, kernel_spatial, output_spatial)) {{ continue; }}
        unsigned int valid_count = 0u;
        for (unsigned int count_kernel = 0u; count_kernel < parameters.geometry[1]; ++count_kernel) {{
            int count_kernel_spatial[3];
            decode_spatial(count_kernel, parameters.kernel, parameters.geometry[0], count_kernel_spatial);
            int count_input_spatial[3];
            if (project_forward(parameters, output_spatial, count_kernel_spatial, count_input_spatial)) {{
                valid_count += 1u;
            }}
        }}
        if (valid_count > 0u) {{
            int output_coordinates[5];
            with_spatial(destination_coordinates[0], destination_coordinates[1], output_spatial,
                parameters.target.rank - 2u, output_coordinates);
            output[physical(parameters.destination, destination_coordinates)] +=
                first[physical(parameters.target, output_coordinates)] / ({scalar})valid_count;
        }}
    }}
"#,
        scalar = scalar
    )
}

fn pooling_backward_maximum(scalar: &str) -> String {
    format!(
        r#"
    int elements = logical_elements(parameters.destination);
    if (linear >= elements) {{ return; }}
    int destination_coordinates[5];
    decode_layout(linear, parameters.destination, destination_coordinates);
    int input_spatial[3] = {{ destination_coordinates[2], destination_coordinates[3], destination_coordinates[4] }};
    for (unsigned int kernel_linear = 0u; kernel_linear < parameters.geometry[1]; ++kernel_linear) {{
        int kernel_spatial[3];
        decode_spatial(kernel_linear, parameters.kernel, parameters.geometry[0], kernel_spatial);
        int output_spatial[3];
        if (!project_inverse(parameters, input_spatial, kernel_spatial, output_spatial)) {{ continue; }}
        {scalar} maximum = ({scalar})0;
        int selected[3] = {{ 0, 0, 0 }};
        bool found = false;
        for (unsigned int candidate_linear = 0u; candidate_linear < parameters.geometry[1]; ++candidate_linear) {{
            int candidate_kernel[3];
            decode_spatial(candidate_linear, parameters.kernel, parameters.geometry[0], candidate_kernel);
            int candidate_spatial[3];
            if (!project_forward(parameters, output_spatial, candidate_kernel, candidate_spatial)) {{ continue; }}
            int candidate_coordinates[5];
            with_spatial(destination_coordinates[0], destination_coordinates[1], candidate_spatial,
                parameters.source.rank - 2u, candidate_coordinates);
            {scalar} candidate = first[physical(parameters.source, candidate_coordinates)];
            if (!found || candidate > maximum) {{
                maximum = candidate;
                for (unsigned int axis = 0u; axis < parameters.geometry[0]; ++axis) {{ selected[axis] = candidate_spatial[axis]; }}
                found = true;
            }}
        }}
        if (found) {{
            bool selected_match = true;
            for (unsigned int axis = 0u; axis < parameters.geometry[0]; ++axis) {{ selected_match = selected_match && selected[axis] == input_spatial[axis]; }}
            if (selected_match) {{
                int output_coordinates[5];
                with_spatial(destination_coordinates[0], destination_coordinates[1], output_spatial,
                    parameters.target.rank - 2u, output_coordinates);
                output[physical(parameters.destination, destination_coordinates)] +=
                    second[physical(parameters.target, output_coordinates)];
            }}
        }}
    }}
"#,
        scalar = scalar
    )
}

fn unfold(scalar: &str) -> String {
    format!(
        r#"
    int elements = logical_elements(parameters.target);
    if (linear >= elements) {{ return; }}
    int output_coordinates[5];
    decode_layout(linear, parameters.target, output_coordinates);
    int channel = output_coordinates[1] / (int)parameters.geometry[1];
    int kernel_linear = output_coordinates[1] % (int)parameters.geometry[1];
    int output_spatial[3];
    decode_spatial(output_coordinates[2], parameters.output_spatial, parameters.geometry[0], output_spatial);
    int kernel_spatial[3];
    decode_spatial(kernel_linear, parameters.kernel, parameters.geometry[0], kernel_spatial);
    int input_spatial[3];
    {scalar} value = ({scalar})0;
    if (project_forward(parameters, output_spatial, kernel_spatial, input_spatial)) {{
        int input_coordinates[5];
        with_spatial(output_coordinates[0], channel, input_spatial,
            parameters.source.rank - 2u, input_coordinates);
        value = first[physical(parameters.source, input_coordinates)];
    }}
    output[physical(parameters.target, output_coordinates)] = value;
"#,
        scalar = scalar
    )
}

fn fold(scalar: &str) -> String {
    format!(
        r#"
    int elements = logical_elements(parameters.destination);
    if (linear >= elements) {{ return; }}
    int destination_coordinates[5];
    decode_layout(linear, parameters.destination, destination_coordinates);
    int input_spatial[3] = {{ destination_coordinates[2], destination_coordinates[3], destination_coordinates[4] }};
    {scalar} value = ({scalar})0;
    for (unsigned int kernel_linear = 0u; kernel_linear < parameters.geometry[1]; ++kernel_linear) {{
        int kernel_spatial[3];
        decode_spatial(kernel_linear, parameters.kernel, parameters.geometry[0], kernel_spatial);
        int output_spatial[3];
        if (!project_inverse(parameters, input_spatial, kernel_spatial, output_spatial)) {{ continue; }}
        int output_coordinates[5];
        output_coordinates[0] = destination_coordinates[0];
        output_coordinates[1] = destination_coordinates[1] * (int)parameters.geometry[1] + (int)kernel_linear;
        output_coordinates[2] = spatial_linear(
            output_spatial, parameters.output_spatial, parameters.geometry[0]);
        value += first[physical(parameters.source, output_coordinates)];
    }}
    second[physical(parameters.destination, destination_coordinates)] = value;
"#,
        scalar = scalar
    )
}

impl WindowDialect for CudaC {
    fn window_source<T: DialectScalar<Self>>(
        operation: WindowOperation,
        width: BlockWidth,
    ) -> String {
        c_family_window_source::<Self, T>(operation, width)
    }
}

impl WindowDialect for HipC {
    fn window_source<T: DialectScalar<Self>>(
        operation: WindowOperation,
        width: BlockWidth,
    ) -> String {
        c_family_window_source::<Self, T>(operation, width)
    }
}
