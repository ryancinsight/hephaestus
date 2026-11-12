//! HIP kernel sources for strided rank-2 axis reductions.

use hephaestus_core::{CombineExpr, HipC, IdentityToken, SumOp};

pub(crate) fn axis_reduction_shader_source<Op: CombineExpr<HipC>, T: IdentityToken<Op, HipC>>()
-> String {
    format!(
        r#"
struct AxisReductionMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

extern "C" __global__ void axis_reduction_kernel(
    AxisReductionMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int axis = meta.offsets[2];
    unsigned int axis_len = (axis == 0u) ? meta.input_shape[0] : meta.input_shape[1];
    unsigned int out_row = (axis == 0u) ? 0u : i;
    unsigned int out_col = (axis == 0u) ? i : 0u;
    {ty} acc = {identity};

    for (unsigned int r = 0u; r < axis_len; r++) {{
        unsigned int in_row = (axis == 0u) ? r : out_row;
        unsigned int in_col = (axis == 0u) ? out_col : r;
        int in_off = (int)meta.offsets[0]
            + (int)in_row * meta.input_strides[0]
            + (int)in_col * meta.input_strides[1];
        {ty} lhs = acc;
        {ty} rhs = input[in_off];
        acc = {expr};
    }}

    int out_off = (int)meta.offsets[1]
        + (int)out_row * meta.output_strides[0]
        + (int)out_col * meta.output_strides[1];
    output[out_off] = acc;
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
        expr = Op::EXPR,
    )
}

pub(crate) fn mean_axis_shader_source<T: IdentityToken<SumOp, HipC>>() -> String {
    format!(
        r#"
struct AxisReductionMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

extern "C" __global__ void mean_axis_kernel(
    AxisReductionMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int axis = meta.offsets[2];
    unsigned int axis_len = (axis == 0u) ? meta.input_shape[0] : meta.input_shape[1];
    unsigned int out_row = (axis == 0u) ? 0u : i;
    unsigned int out_col = (axis == 0u) ? i : 0u;
    {ty} acc = {identity};

    for (unsigned int r = 0u; r < axis_len; r++) {{
        unsigned int in_row = (axis == 0u) ? r : out_row;
        unsigned int in_col = (axis == 0u) ? out_col : r;
        int in_off = (int)meta.offsets[0]
            + (int)in_row * meta.input_strides[0]
            + (int)in_col * meta.input_strides[1];
        acc = acc + input[in_off];
    }}

    int out_off = (int)meta.offsets[1]
        + (int)out_row * meta.output_strides[0]
        + (int)out_col * meta.output_strides[1];
    output[out_off] = acc / ({ty})axis_len;
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
    )
}
