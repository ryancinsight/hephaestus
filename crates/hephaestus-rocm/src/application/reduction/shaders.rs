//! HIP kernel sources for contiguous ROCm reductions.

use hephaestus_core::{BlockWidth, CombineExpr, HipC, IdentityToken};

pub(crate) fn shader_source<Op: CombineExpr<HipC>, T: IdentityToken<Op, HipC>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"
extern "C" __global__ void reduction_kernel(
    const {ty}* input,
    {ty}* output,
    unsigned int n
) {{
    extern __shared__ {ty} shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = i < n ? input[i] : {identity};
    __syncthreads();

    for (unsigned int stride = {wg}u / 2u; stride > 0u; stride /= 2u) {{
        if (tid < stride) {{
            {ty} lhs = shared_data[tid];
            {ty} rhs = shared_data[tid + stride];
            shared_data[tid] = {expr};
        }}
        __syncthreads();
    }}

    if (tid == 0u) {{
        output[blockIdx.x] = shared_data[0];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = T::TOKEN,
        expr = Op::EXPR,
    )
}
