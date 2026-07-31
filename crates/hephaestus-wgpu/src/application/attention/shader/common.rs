pub(super) fn prelude(width: u32) -> String {
    format!(
        r#"
struct LayoutMeta {{
    shape: vec4<u32>,
    strides: vec4<i32>,
    offset: vec4<i32>,
}}

struct AttentionMeta {{
    query: LayoutMeta,
    key: LayoutMeta,
    value: LayoutMeta,
    weights: LayoutMeta,
    grad_output: LayoutMeta,
    destination: LayoutMeta,
    keep_mask: LayoutMeta,
    dimensions: vec4<u32>,
    value_and_flags: vec4<u32>,
    scale_and_padding: vec4<f32>,
}}

fn physical(metadata: LayoutMeta, first: u32, second: u32, third: u32) -> u32 {{
    return u32(
        metadata.offset.x +
        i32(first) * metadata.strides.x +
        i32(second) * metadata.strides.y +
        i32(third) * metadata.strides.z
    );
}}

const WORKGROUP_WIDTH: u32 = {width}u;
"#
    )
}
