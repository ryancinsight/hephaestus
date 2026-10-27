//! CUDA map-reduction metadata, expression markers, and kernel source builder.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, CombineExpr, CudaC, DialectScalar, IdentityToken, MaxOp, SumOp};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MapReductionMeta {
    pub(super) shape: [u32; 8],
    pub(super) a_strides: [i32; 8],
    pub(super) b_strides: [i32; 8],
    pub(super) offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<MapReductionMeta>() == 112);

pub(crate) trait MapReductionOp: Copy + Send + Sync + 'static {
    type ReduceOp: CombineExpr<CudaC>;

    const EXPR: &'static str;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IdentityMap;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DotMap;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AbsMap;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SquareMap;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MaxAbsMap;

impl MapReductionOp for IdentityMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "lhs";
}

impl MapReductionOp for DotMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "lhs * rhs";
}

impl MapReductionOp for AbsMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "abs(lhs)";
}

impl MapReductionOp for SquareMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "lhs * rhs";
}

impl MapReductionOp for MaxAbsMap {
    type ReduceOp = MaxOp;

    const EXPR: &'static str = "abs(lhs)";
}

pub(crate) fn shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: MapReductionOp,
    T: DialectScalar<CudaC> + IdentityToken<Op::ReduceOp, CudaC>,
{
    format!(
        r#"
struct MapReductionMeta {{
    unsigned int shape[8];
    int a_strides[8];
    int b_strides[8];
    unsigned int offsets[4];
}};

extern "C" __global__ void map_reduction_kernel(
    MapReductionMeta meta,
    const {ty}* a,
    const {ty}* b,
    {ty}* output
) {{
    extern __shared__ {ty} shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    {ty} value = {identity};
    if (i < meta.offsets[3]) {{
        unsigned int rem = i;
        int a_off = (int)meta.offsets[0];
        int b_off = (int)meta.offsets[1];
        for (int dimension = 7; dimension >= 0; --dimension) {{
            unsigned int dim = meta.shape[dimension];
            unsigned int index = rem % dim;
            rem /= dim;
            a_off += (int)index * meta.a_strides[dimension];
            b_off += (int)index * meta.b_strides[dimension];
        }}
        {ty} lhs = a[a_off];
        {ty} rhs = b[b_off];
        value = {expr};
    }}
    shared_data[tid] = value;
    __syncthreads();

    for (unsigned int stride = {width}u / 2u; stride > 0u; stride /= 2u) {{
        if (tid < stride) {{
            {ty} lhs = shared_data[tid];
            {ty} rhs = shared_data[tid + stride];
            shared_data[tid] = {reduce};
        }}
        __syncthreads();
    }}

    if (tid == 0u) {{
        output[blockIdx.x] = shared_data[0];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = <T as IdentityToken<Op::ReduceOp, CudaC>>::TOKEN,
        expr = Op::EXPR,
        width = width.get(),
        reduce = <Op::ReduceOp as CombineExpr<CudaC>>::EXPR,
    )
}

#[cfg(test)]
mod tests {
    use super::{DotMap, shader_source};
    use hephaestus_core::BlockWidth;

    #[test]
    fn source_declares_strided_map_reduction_contract() {
        let source = shader_source::<DotMap, f32>(BlockWidth::DEFAULT);
        assert!(source.contains("shape[8]"));
        assert!(source.contains("a_strides[8]"));
        assert!(source.contains("b_strides[8]"));
        assert!(source.contains("shared_data"));
        assert!(source.contains("lhs * rhs"));
        assert!(source.contains("__syncthreads();"));
    }
}
