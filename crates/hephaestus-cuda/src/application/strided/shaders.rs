//! Strided elementwise shader builders.

use super::types::{CUDA_DECODE, CUDA_META};
use hephaestus_core::{BinaryExpr, CudaC, DialectScalar, UnaryExpr};

pub(crate) fn binary_shader<T: DialectScalar<CudaC>>(expr: &'static str) -> String {
    format!(
        r#"{prelude}
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
    {ty} lhs = lhs_ptr[a_off];
    {ty} rhs = rhs_ptr[b_off];
    out[o_off] = {expr};
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_META,
        ty = T::TYPE_TOKEN,
        decode = CUDA_DECODE,
        expr = expr,
    )
}

pub(crate) fn unary_shader<Op: UnaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}
{meta}
extern "C" __global__ void unary_strided_kernel(
    Meta lmeta,
    const {ty}* a,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} x = a[a_off];
    out[o_off] = {expr};
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_META,
        ty = T::TYPE_TOKEN,
        decode = CUDA_DECODE,
        expr = Op::EXPR,
    )
}

pub(crate) fn scalar_shader<Op: BinaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}
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
    {ty} lhs = input[a_off];
    {ty} rhs = scalar;
    out[o_off] = {expr};
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_META,
        ty = T::TYPE_TOKEN,
        decode = CUDA_DECODE,
        expr = Op::EXPR,
    )
}
