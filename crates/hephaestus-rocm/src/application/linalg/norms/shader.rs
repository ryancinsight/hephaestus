//! HIP map-reduction metadata, expression markers, and kernel source builder.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, HephaestusError, HipC, IdentityToken, MaxOp, Result,
    SumOp,
};

const MAX_STRIDED_RANK: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MapReductionMeta {
    pub(super) shape: [u32; 4],
    pub(super) a_strides: [i32; 4],
    pub(super) b_strides: [i32; 4],
    pub(super) offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<MapReductionMeta>() == 64);

pub(crate) trait MapReductionOp: Copy + Send + Sync + 'static {
    type ReduceOp: CombineExpr<HipC>;

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
    T: DialectScalar<HipC> + IdentityToken<Op::ReduceOp, HipC>,
{
    format!(
        r#"
struct MapReductionMeta {{
    unsigned int shape[4];
    int a_strides[4];
    int b_strides[4];
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
        for (int dimension = 3; dimension >= 0; dimension--) {{
            unsigned int dim = meta.shape[dimension];
            unsigned int index = rem % dim;
            rem = rem / dim;
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
        identity = <T as IdentityToken<Op::ReduceOp, HipC>>::TOKEN,
        expr = Op::EXPR,
        width = width.get(),
        reduce = <Op::ReduceOp as CombineExpr<HipC>>::EXPR,
    )
}

pub(super) fn map_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 4]> {
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "ROCm strided reductions support rank <= 4"
        );
    }
    let mut mapped = [1_u32; 4];
    for (dimension, &extent) in shape.iter().enumerate() {
        mapped[4 - N + dimension] =
            u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dimension {extent} exceeds u32 range"),
            })?;
    }
    Ok(mapped)
}

pub(super) fn map_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 4]> {
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "ROCm strided reductions support rank <= 4"
        );
    }
    let mut mapped = [0_i32; 4];
    for (dimension, &stride) in strides.iter().enumerate() {
        mapped[4 - N + dimension] =
            i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("stride {stride} exceeds i32 range"),
            })?;
    }
    Ok(mapped)
}
