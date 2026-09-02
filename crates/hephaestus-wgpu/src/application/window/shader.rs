use hephaestus_core::PoolingMode;

const PRELUDE: &str = r#"
struct WindowLayoutMeta {
    shape: array<vec4<u32>, 2>,
    strides: array<vec4<i32>, 2>,
    offset_and_rank: vec4<u32>,
}

struct WindowMeta {
    source: WindowLayoutMeta,
    target_layout: WindowLayoutMeta,
    destination: WindowLayoutMeta,
    kernel: vec4<u32>,
    stride: vec4<u32>,
    padding_values: vec4<u32>,
    dilation: vec4<u32>,
    output_spatial: vec4<u32>,
    geometry: vec4<u32>,
}

fn extent(view_layout: WindowLayoutMeta, axis: u32) -> u32 {
    return view_layout.shape[axis / 4u][axis % 4u];
}

fn stride(view_layout: WindowLayoutMeta, axis: u32) -> i32 {
    return view_layout.strides[axis / 4u][axis % 4u];
}

fn physical(view_layout: WindowLayoutMeta, coordinates: array<u32, 5>) -> u32 {
    var position = i32(view_layout.offset_and_rank.x);
    var axis = 0u;
    loop {
        if (axis >= view_layout.offset_and_rank.y) { break; }
        position += i32(coordinates[axis]) * stride(view_layout, axis);
        axis += 1u;
    }
    return u32(position);
}

fn logical_elements(view_layout: WindowLayoutMeta) -> u32 {
    var elements = 1u;
    var axis = 0u;
    loop {
        if (axis >= view_layout.offset_and_rank.y) { break; }
        elements *= extent(view_layout, axis);
        axis += 1u;
    }
    return elements;
}

fn decode_layout(linear_input: u32, view_layout: WindowLayoutMeta) -> array<u32, 5> {
    var linear = linear_input;
    var coordinates: array<u32, 5>;
    var axis = i32(view_layout.offset_and_rank.y) - 1;
    loop {
        if (axis < 0) { break; }
        let axis_extent = extent(view_layout, u32(axis));
        coordinates[u32(axis)] = linear % axis_extent;
        linear /= axis_extent;
        axis -= 1;
    }
    return coordinates;
}

fn decode_spatial(linear_input: u32, extents: vec4<u32>, rank: u32) -> array<u32, 3> {
    var linear = linear_input;
    var coordinates: array<u32, 3>;
    var axis = i32(rank) - 1;
    loop {
        if (axis < 0) { break; }
        let extent = extents[u32(axis)];
        coordinates[u32(axis)] = linear % extent;
        linear /= extent;
        axis -= 1;
    }
    return coordinates;
}

fn decode_kernel(linear_input: u32, parameters: WindowMeta) -> array<u32, 3> {
    return decode_spatial(linear_input, parameters.kernel, parameters.geometry.x);
}

fn spatial_linear(spatial: array<u32, 3>, extents: vec4<u32>, rank: u32) -> u32 {
    var linear = 0u;
    var axis = 0u;
    loop {
        if (axis >= rank) { break; }
        linear = linear * extents[axis] + spatial[axis];
        axis += 1u;
    }
    return linear;
}

fn with_spatial(
    batch: u32,
    channel: u32,
    spatial: array<u32, 3>,
    rank: u32,
) -> array<u32, 5> {
    var coordinates: array<u32, 5>;
    coordinates[0] = batch;
    coordinates[1] = channel;
    var axis = 0u;
    loop {
        if (axis >= rank) { break; }
        coordinates[axis + 2u] = spatial[axis];
        axis += 1u;
    }
    return coordinates;
}
"#;

pub(super) fn pooling_forward<T: hephaestus_core::DialectScalar<hephaestus_core::Wgsl>>(
    mode: PoolingMode,
    width: u32,
) -> String {
    let scalar = T::TYPE_TOKEN;
    let reduction = match mode {
        PoolingMode::Maximum => format!(
            r#"
    var value: {scalar} = {scalar}(0);
    var found = false;
    var kernel_linear = 0u;
    loop {{
        if (kernel_linear >= parameters.geometry.y) {{ break; }}
        let kernel_spatial = decode_kernel(kernel_linear, parameters);
        var input_spatial: array<u32, 3>;
        var valid = true;
        var axis = 0u;
        loop {{
            if (axis >= parameters.geometry.x) {{ break; }}
            let projected = i32(output_spatial[axis]) * i32(parameters.stride[axis]) +
                i32(kernel_spatial[axis]) * i32(parameters.dilation[axis]) -
                i32(parameters.padding_values[axis]);
            if (projected < 0 || projected >= i32(extent(parameters.source, axis + 2u))) {{
                valid = false;
            }} else {{
                input_spatial[axis] = u32(projected);
            }}
            axis += 1u;
        }}
        if (valid) {{
            let input_coordinates = with_spatial(
                output_coordinates[0], output_coordinates[1], input_spatial,
                parameters.source.offset_and_rank.y - 2u,
            );
            let candidate = input[physical(parameters.source, input_coordinates)];
            if (!found || candidate > value) {{
                value = candidate;
                found = true;
            }}
        }}
        kernel_linear += 1u;
    }}
"#,
        ),
        PoolingMode::Average => format!(
            r#"
    var value: {scalar} = {scalar}(0);
    var valid_count = 0u;
    var kernel_linear = 0u;
    loop {{
        if (kernel_linear >= parameters.geometry.y) {{ break; }}
        let kernel_spatial = decode_kernel(kernel_linear, parameters);
        var input_spatial: array<u32, 3>;
        var valid = true;
        var axis = 0u;
        loop {{
            if (axis >= parameters.geometry.x) {{ break; }}
            let projected = i32(output_spatial[axis]) * i32(parameters.stride[axis]) +
                i32(kernel_spatial[axis]) * i32(parameters.dilation[axis]) -
                i32(parameters.padding_values[axis]);
            if (projected < 0 || projected >= i32(extent(parameters.source, axis + 2u))) {{
                valid = false;
            }} else {{
                input_spatial[axis] = u32(projected);
            }}
            axis += 1u;
        }}
        if (valid) {{
            let input_coordinates = with_spatial(
                output_coordinates[0], output_coordinates[1], input_spatial,
                parameters.source.offset_and_rank.y - 2u,
            );
            value += input[physical(parameters.source, input_coordinates)];
            valid_count += 1u;
        }}
        kernel_linear += 1u;
    }}
    if (valid_count > 0u) {{ value /= {scalar}(valid_count); }}
"#,
        ),
    };
    format!(
        r#"{PRELUDE}
@group(0) @binding(0) var<storage, read> input: array<{scalar}>;
@group(0) @binding(1) var<storage, read_write> output: array<{scalar}>;
@group(0) @binding(2) var<uniform> parameters: WindowMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = logical_elements(parameters.target_layout);
    if (id.x >= elements) {{ return; }}
    let output_coordinates = decode_layout(id.x, parameters.target_layout);
    let output_spatial = array<u32, 3>(
        output_coordinates[2], output_coordinates[3], output_coordinates[4]);
{reduction}
    output[physical(parameters.target_layout, output_coordinates)] = value;
}}
"#,
        PRELUDE = PRELUDE,
        scalar = scalar,
        width = width,
        reduction = reduction,
    )
}

pub(super) fn pooling_backward<T: hephaestus_core::DialectScalar<hephaestus_core::Wgsl>>(
    mode: PoolingMode,
    width: u32,
) -> String {
    let scalar = T::TYPE_TOKEN;
    let input_binding = match mode {
        PoolingMode::Maximum => {
            format!("@group(0) @binding(0) var<storage, read> input: array<{scalar}>;")
        }
        PoolingMode::Average => String::new(),
    };
    let grad_binding = u32::from(matches!(mode, PoolingMode::Maximum));
    let destination_binding = grad_binding + 1;
    let reduction = match mode {
        PoolingMode::Average => format!(
            r#"
        var valid_count = 0u;
        var count_kernel = 0u;
        loop {{
            if (count_kernel >= parameters.geometry.y) {{ break; }}
            let count_kernel_spatial = decode_kernel(count_kernel, parameters);
            var count_valid = true;
            var count_axis = 0u;
            loop {{
                if (count_axis >= parameters.geometry.x) {{ break; }}
                let count_projected = i32(output_spatial[count_axis]) *
                    i32(parameters.stride[count_axis]) +
                    i32(count_kernel_spatial[count_axis]) *
                    i32(parameters.dilation[count_axis]) -
                    i32(parameters.padding_values[count_axis]);
                count_valid = count_valid && count_projected >= 0 &&
                    count_projected < i32(extent(parameters.source, count_axis + 2u));
                count_axis += 1u;
            }}
            if (count_valid) {{ valid_count += 1u; }}
            count_kernel += 1u;
        }}
        if (valid_count > 0u) {{
            grad_input[physical(parameters.destination, destination_coordinates)] +=
                grad_output[physical(parameters.target_layout, output_coordinates)] /
                {scalar}(valid_count);
        }}
"#
        ),
        PoolingMode::Maximum => format!(
            r#"
        var maximum: {scalar} = {scalar}(0);
        var selected: array<u32, 3>;
        var found = false;
        var max_kernel = 0u;
        loop {{
            if (max_kernel >= parameters.geometry.y) {{ break; }}
            let max_kernel_spatial = decode_kernel(max_kernel, parameters);
            var max_input_spatial: array<u32, 3>;
            var max_valid = true;
            var max_axis = 0u;
            loop {{
                if (max_axis >= parameters.geometry.x) {{ break; }}
                let max_projected = i32(output_spatial[max_axis]) *
                    i32(parameters.stride[max_axis]) +
                    i32(max_kernel_spatial[max_axis]) *
                    i32(parameters.dilation[max_axis]) -
                    i32(parameters.padding_values[max_axis]);
                if (max_projected < 0 ||
                    max_projected >= i32(extent(parameters.source, max_axis + 2u))) {{
                    max_valid = false;
                }} else {{
                    max_input_spatial[max_axis] = u32(max_projected);
                }}
                max_axis += 1u;
            }}
            if (max_valid) {{
                let max_coordinates = with_spatial(
                    output_coordinates[0], output_coordinates[1], max_input_spatial,
                    parameters.source.offset_and_rank.y - 2u,
                );
                let candidate = input[physical(parameters.source, max_coordinates)];
                if (!found || candidate > maximum) {{
                    maximum = candidate;
                    selected = max_input_spatial;
                    found = true;
                }}
            }}
            max_kernel += 1u;
        }}
        var selected_matches = found;
        var selected_axis = 0u;
        loop {{
            if (selected_axis >= parameters.geometry.x) {{ break; }}
            selected_matches = selected_matches &&
                selected[selected_axis] == input_spatial[selected_axis];
            selected_axis += 1u;
        }}
        if (selected_matches) {{
            grad_input[physical(parameters.destination, destination_coordinates)] +=
                grad_output[physical(parameters.target_layout, output_coordinates)];
        }}
"#,
        ),
    };
    format!(
        r#"{PRELUDE}
{input_binding}
@group(0) @binding({grad_binding}) var<storage, read> grad_output: array<{scalar}>;
@group(0) @binding({destination_binding}) var<storage, read_write> grad_input: array<{scalar}>;
@group(0) @binding({metadata_binding}) var<uniform> parameters: WindowMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = logical_elements(parameters.destination);
    if (id.x >= elements) {{ return; }}
    let destination_coordinates = decode_layout(id.x, parameters.destination);
    var input_spatial: array<u32, 3>;
    var input_axis = 0u;
    loop {{
        if (input_axis >= parameters.geometry.x) {{ break; }}
        input_spatial[input_axis] = destination_coordinates[input_axis + 2u];
        input_axis += 1u;
    }}
    var kernel_linear = 0u;
    loop {{
        if (kernel_linear >= parameters.geometry.y) {{ break; }}
        let kernel_spatial = decode_kernel(kernel_linear, parameters);
        var output_spatial: array<u32, 3>;
        var valid = true;
        var axis = 0u;
        loop {{
            if (axis >= parameters.geometry.x) {{ break; }}
            let numerator = i32(input_spatial[axis]) +
                i32(parameters.padding_values[axis]) -
                i32(kernel_spatial[axis]) * i32(parameters.dilation[axis]);
            let axis_stride = i32(parameters.stride[axis]);
            if (numerator < 0 || numerator % axis_stride != 0) {{
                valid = false;
            }} else {{
                let projected = numerator / axis_stride;
                if (projected >= i32(parameters.output_spatial[axis])) {{
                    valid = false;
                }} else {{
                    output_spatial[axis] = u32(projected);
                }}
            }}
            axis += 1u;
        }}
        if (valid) {{
            let output_coordinates = with_spatial(
                destination_coordinates[0], destination_coordinates[1], output_spatial,
                parameters.target_layout.offset_and_rank.y - 2u,
            );
{reduction}
        }}
        kernel_linear += 1u;
    }}
}}
"#,
        PRELUDE = PRELUDE,
        input_binding = input_binding,
        grad_binding = grad_binding,
        destination_binding = destination_binding,
        metadata_binding = destination_binding + 1,
        scalar = scalar,
        width = width,
        reduction = reduction,
    )
}

pub(super) fn sliding_window<T: hephaestus_core::DialectScalar<hephaestus_core::Wgsl>>(
    unfold: bool,
    width: u32,
) -> String {
    let scalar = T::TYPE_TOKEN;
    if unfold {
        format!(
            r#"{PRELUDE}
@group(0) @binding(0) var<storage, read> input: array<{scalar}>;
@group(0) @binding(1) var<storage, read_write> output: array<{scalar}>;
@group(0) @binding(2) var<uniform> parameters: WindowMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = logical_elements(parameters.target_layout);
    if (id.x >= elements) {{ return; }}
    let output_coordinates = decode_layout(id.x, parameters.target_layout);
    let channel = output_coordinates[1] / parameters.geometry.y;
    let kernel_linear = output_coordinates[1] % parameters.geometry.y;
    let output_spatial = decode_spatial(
        output_coordinates[2], parameters.output_spatial, parameters.geometry.x);
    let kernel_spatial = decode_kernel(kernel_linear, parameters);
    var input_spatial: array<u32, 3>;
    var valid = true;
    var axis = 0u;
    loop {{
        if (axis >= parameters.geometry.x) {{ break; }}
        let projected = i32(output_spatial[axis]) * i32(parameters.stride[axis]) +
            i32(kernel_spatial[axis]) * i32(parameters.dilation[axis]) -
            i32(parameters.padding_values[axis]);
        if (projected < 0 || projected >= i32(extent(parameters.source, axis + 2u))) {{
            valid = false;
        }} else {{
            input_spatial[axis] = u32(projected);
        }}
        axis += 1u;
    }}
    var value: {scalar} = {scalar}(0);
    if (valid) {{
        let input_coordinates = with_spatial(
            output_coordinates[0], channel, input_spatial,
            parameters.source.offset_and_rank.y - 2u,
        );
        value = input[physical(parameters.source, input_coordinates)];
    }}
    output[physical(parameters.target_layout, output_coordinates)] = value;
}}
"#,
            PRELUDE = PRELUDE,
            scalar = scalar,
            width = width,
        )
    } else {
        format!(
            r#"{PRELUDE}
@group(0) @binding(0) var<storage, read> input: array<{scalar}>;
@group(0) @binding(1) var<storage, read_write> output: array<{scalar}>;
@group(0) @binding(2) var<uniform> parameters: WindowMeta;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let elements = logical_elements(parameters.target_layout);
    if (id.x >= elements) {{ return; }}
    let output_coordinates = decode_layout(id.x, parameters.target_layout);
    let output_spatial = array<u32, 3>(
        output_coordinates[2], output_coordinates[3], output_coordinates[4]);
    var value: {scalar} = {scalar}(0);
    var kernel_linear = 0u;
    loop {{
        if (kernel_linear >= parameters.geometry.y) {{ break; }}
        let kernel_spatial = decode_kernel(kernel_linear, parameters);
        var column_spatial: array<u32, 3>;
        var valid = true;
        var axis = 0u;
        loop {{
            if (axis >= parameters.geometry.x) {{ break; }}
            let numerator = i32(output_spatial[axis]) +
                i32(parameters.padding_values[axis]) -
                i32(kernel_spatial[axis]) * i32(parameters.dilation[axis]);
            let axis_stride = i32(parameters.stride[axis]);
            if (numerator < 0 || numerator % axis_stride != 0) {{
                valid = false;
            }} else {{
                let projected = numerator / axis_stride;
                if (projected >= i32(parameters.output_spatial[axis])) {{
                    valid = false;
                }} else {{
                    column_spatial[axis] = u32(projected);
                }}
            }}
            axis += 1u;
        }}
        if (valid) {{
            let column_coordinates = array<u32, 5>(
                output_coordinates[0], output_coordinates[1] * parameters.geometry.y + kernel_linear,
                spatial_linear(column_spatial, parameters.output_spatial, parameters.geometry.x),
                0u, 0u);
            value += input[physical(parameters.source, column_coordinates)];
        }}
        kernel_linear += 1u;
    }}
    output[physical(parameters.target_layout, output_coordinates)] = value;
}}
"#,
            PRELUDE = PRELUDE,
            scalar = scalar,
            width = width,
        )
    }
}
