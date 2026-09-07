//! Shared HIP metadata declaration and strided elementwise kernels.

use super::metadata::MAX_STRIDED_RANK;
use hephaestus_core::{BinaryExpr, DialectScalar, HipC, UnaryExpr};

#[cfg(test)]
mod tests;

pub(crate) fn hip_meta() -> String {
    format!(
        r#"
struct Meta {{
    unsigned int shape[{rank}];
    int a_strides[{rank}];
    int b_strides[{rank}];
    int out_strides[{rank}];
    unsigned int offsets[4];
}};
"#,
        rank = MAX_STRIDED_RANK
    )
}

pub(crate) fn hip_decode() -> String {
    // For a nonempty dispatch, P = product(shape) <= u32::MAX and
    // sum(shape[d] - 1) <= P - 1. Every partial address sum is bounded by
    // u32::MAX + (P - 1) * 2^31 <= i64::MAX, including negative i32 strides.
    // Widen before multiplying: valid sparse layouts can exceed i32 offsets.
    format!(
        r#"
    unsigned int rem = i;
    long long a_offset = (long long)lmeta.offsets[0];
    long long b_offset = (long long)lmeta.offsets[1];
    long long out_offset = (long long)lmeta.offsets[2];
    for (int dimension = {last_axis}; dimension >= 0; dimension--) {{
        unsigned int extent = lmeta.shape[dimension];
        long long index = (long long)(rem % extent);
        rem = rem / extent;
        a_offset += index * (long long)lmeta.a_strides[dimension];
        b_offset += index * (long long)lmeta.b_strides[dimension];
        out_offset += index * (long long)lmeta.out_strides[dimension];
    }}
"#,
        last_axis = MAX_STRIDED_RANK - 1
    )
}

pub(crate) fn binary_shader<T: DialectScalar<HipC>>(expr: &'static str) -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void binary_strided_kernel(
    Meta lmeta,
    const {ty}* lhs_ptr,
    const {ty}* rhs_ptr,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} lhs = lhs_ptr[a_offset];
    {ty} rhs = rhs_ptr[b_offset];
    out[out_offset] = {expr};
}}
"#,
        meta = hip_meta(),
        decode = hip_decode(),
        ty = T::TYPE_TOKEN,
        expr = expr,
    )
}

pub(crate) fn unary_shader<Op: UnaryExpr<HipC>, T: DialectScalar<HipC>>() -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void unary_strided_kernel(
    Meta lmeta,
    const {ty}* input,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} x = input[a_offset];
    out[out_offset] = {expr};
}}
"#,
        meta = hip_meta(),
        decode = hip_decode(),
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
    )
}

pub(crate) fn scalar_shader<Op: BinaryExpr<HipC>, T: DialectScalar<HipC>>() -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void scalar_strided_kernel(
    Meta lmeta,
    const {ty}* input,
    {ty} scalar,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} lhs = input[a_offset];
    {ty} rhs = scalar;
    out[out_offset] = {expr};
}}
"#,
        meta = hip_meta(),
        decode = hip_decode(),
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
    )
}
