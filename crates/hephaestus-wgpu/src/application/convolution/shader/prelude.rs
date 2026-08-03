pub(super) fn prelude(scalar: &str) -> String {
    format!(
        r#"
struct LayoutMeta {{
    shape_low: vec4<u32>,
    shape_high: vec4<u32>,
    strides_low: vec4<i32>,
    strides_high: vec4<i32>,
    offset_and_rank: vec4<u32>,
}}

struct ConvolutionMeta {{
    input: LayoutMeta,
    weight: LayoutMeta,
    output: LayoutMeta,
    destination: LayoutMeta,
    stride_low: vec4<u32>,
    stride_high: vec4<u32>,
    padding_low: vec4<u32>,
    padding_high: vec4<u32>,
    dilation_low: vec4<u32>,
    dilation_high: vec4<u32>,
    output_padding_low: vec4<u32>,
    output_padding_high: vec4<u32>,
    spatial_rank_and_flags: vec4<u32>,
}}

fn layout_extent(descriptor: LayoutMeta, axis: u32) -> u32 {{
    if (axis < 4u) {{
        return descriptor.shape_low[axis];
    }}
    return descriptor.shape_high[axis - 4u];
}}

fn layout_stride(descriptor: LayoutMeta, axis: u32) -> i32 {{
    if (axis < 4u) {{
        return descriptor.strides_low[axis];
    }}
    return descriptor.strides_high[axis - 4u];
}}

fn parameter(low: vec4<u32>, high: vec4<u32>, axis: u32) -> u32 {{
    if (axis < 4u) {{
        return low[axis];
    }}
    return high[axis - 4u];
}}

fn physical(descriptor: LayoutMeta, coordinates: array<u32, 5>) -> u32 {{
    var position = i32(descriptor.offset_and_rank.x);
    var axis = 0u;
    loop {{
        if (axis >= descriptor.offset_and_rank.y) {{
            break;
        }}
        position += i32(coordinates[axis]) * layout_stride(descriptor, axis);
        axis += 1u;
    }}
    return u32(position);
}}

fn decode_layout(linear_input: u32, descriptor: LayoutMeta) -> array<u32, 5> {{
    var linear = linear_input;
    var coordinates: array<u32, 5>;
    var axis = i32(descriptor.offset_and_rank.y) - 1;
    loop {{
        if (axis < 0) {{
            break;
        }}
        let extent = layout_extent(descriptor, u32(axis));
        coordinates[u32(axis)] = linear % extent;
        linear /= extent;
        axis -= 1;
    }}
    return coordinates;
}}

fn spatial_elements(descriptor: LayoutMeta) -> u32 {{
    var elements = 1u;
    var axis = 2u;
    loop {{
        if (axis >= descriptor.offset_and_rank.y) {{
            break;
        }}
        elements *= layout_extent(descriptor, axis);
        axis += 1u;
    }}
    return elements;
}}

fn decode_spatial(linear_input: u32, descriptor: LayoutMeta) -> array<u32, 3> {{
    var linear = linear_input;
    var coordinates: array<u32, 3>;
    var axis = i32(descriptor.offset_and_rank.y) - 1;
    loop {{
        if (axis < 2) {{
            break;
        }}
        let extent = layout_extent(descriptor, u32(axis));
        coordinates[u32(axis) - 2u] = linear % extent;
        linear /= extent;
        axis -= 1;
    }}
    return coordinates;
}}

fn coordinates_with_spatial(
    batch: u32,
    channel: u32,
    spatial: array<u32, 3>,
    rank: u32,
) -> array<u32, 5> {{
    var coordinates: array<u32, 5>;
    coordinates[0] = batch;
    coordinates[1] = channel;
    var axis = 2u;
    loop {{
        if (axis >= rank) {{
            break;
        }}
        coordinates[axis] = spatial[axis - 2u];
        axis += 1u;
    }}
    return coordinates;
}}

fn zero() -> {scalar} {{
    return {scalar}(0);
}}
"#
    )
}
