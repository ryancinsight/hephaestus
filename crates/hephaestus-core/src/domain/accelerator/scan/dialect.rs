use crate::domain::dialect::{CudaC, DialectScalar, HipC, KernelDialect};
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, IdentityToken};

/// A kernel dialect that can express a rank-2 axis scan.
///
/// Source generation belongs to the dialect, not the device: every backend
/// compiling HIP C++ wants byte-identical HIP C++. The trait is open, so a
/// backend crate introducing its own dialect supplies its own scan source
/// without an upstream edit.
pub trait AxisScanDialect: KernelDialect {
    /// Emit the scan kernel for combining operator `Op` over scalar `T` at
    /// block width `width`. The entry point is named [`AXIS_SCAN_ENTRY`].
    fn axis_scan_source<Op, T>(width: BlockWidth) -> String
    where
        Op: CombineExpr<Self>,
        T: IdentityToken<Op, Self>;
}

/// Entry-point name every [`AxisScanDialect`] implementation must emit.
pub const AXIS_SCAN_ENTRY: &str = "scan_kernel";

/// C-family (CUDA C++ / HIP C++) axis-scan kernel source.
///
/// One block owns one scan line. Each lane folds a contiguous chunk in
/// logical order, then lane zero folds the chunk totals in order, and a second
/// pass applies the prefix of preceding chunks to each local prefix. This is
/// the tiled scan theorem: for associative `Op::EXPR` every element receives
/// the same mathematical fold as a sequential scan. Floating-point
/// reassociation is explicit and is covered by the provider's derived error
/// bound.
///
/// NVRTC and hipRTC accept the same text here, so both dialects delegate to
/// this function rather than carrying a copy.
#[must_use]
pub fn c_family_axis_scan_source<L, Op, T>(width: BlockWidth) -> String
where
    L: KernelDialect,
    Op: CombineExpr<L>,
    T: IdentityToken<Op, L>,
{
    format!(
        r#"{prelude}
struct AxisScanMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

__device__ unsigned int source_offset(AxisScanMeta meta, unsigned int row, unsigned int col) {{
    int off = (int)meta.offsets[0]
        + (int)row * meta.input_strides[0]
        + (int)col * meta.input_strides[1];
    return (unsigned int)off;
}}

__device__ unsigned int dest_offset(AxisScanMeta meta, unsigned int row, unsigned int col) {{
    int off = (int)meta.offsets[1]
        + (int)row * meta.output_strides[0]
        + (int)col * meta.output_strides[1];
    return (unsigned int)off;
}}

// One block owns one scan line. Each lane folds a contiguous chunk in logical
// order, then lane zero folds chunk totals in order. The second pass applies
// the prefix of preceding chunks to each local prefix. This is the tiled scan
// theorem: every element receives the same mathematical fold as a sequential
// scan for associative `expr`; floating-point reassociation is explicit and
// is covered by the provider's derived error bound.
extern __shared__ {ty} partial[];

extern "C" __global__ void {entry}(
    AxisScanMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int line = blockIdx.x;
    unsigned int lane = threadIdx.x;

    unsigned int rows = meta.input_shape[0];
    unsigned int cols = meta.input_shape[1];
    unsigned int axis = meta.offsets[2] & 1u;
    bool reverse = (meta.offsets[2] & 2u) != 0u;
    unsigned int len = (axis == 0u) ? rows : cols;
    unsigned int chunk_len = (len + {wg}u - 1u) / {wg}u;
    unsigned int start = lane * chunk_len;
    unsigned int end = min(start + chunk_len, len);
    {ty} local_acc = {identity};

    // Empty lanes retain the identity. `axis` and `reverse` are uniform
    // across the launch, so only the loop bounds vary by lane.
    for (unsigned int s = start; s < end; s++) {{
        unsigned int idx = reverse ? (len - 1u - s) : s;
        unsigned int row = (axis == 0u) ? idx : line;
        unsigned int col = (axis == 0u) ? line : idx;
        {ty} lhs = local_acc;
        {ty} rhs = input[source_offset(meta, row, col)];
        local_acc = {expr};
        output[dest_offset(meta, row, col)] = local_acc;
    }}
    partial[lane] = local_acc;
    __syncthreads();

    if (lane == 0u) {{
        {ty} prefix = {identity};
        for (unsigned int chunk = 0u; chunk < {wg}u; chunk++) {{
            {ty} total = partial[chunk];
            partial[chunk] = prefix;
            {ty} lhs = prefix;
            {ty} rhs = total;
            prefix = {expr};
        }}
    }}
    __syncthreads();

    {ty} prefix = partial[lane];
    for (unsigned int s = start; s < end; s++) {{
        unsigned int idx = reverse ? (len - 1u - s) : s;
        unsigned int row = (axis == 0u) ? idx : line;
        unsigned int col = (axis == 0u) ? line : idx;
        {ty} lhs = prefix;
        {ty} rhs = output[dest_offset(meta, row, col)];
        output[dest_offset(meta, row, col)] = {expr};
    }}
}}
"#,
        prelude = <T as DialectScalar<L>>::PRELUDE,
        ty = <T as DialectScalar<L>>::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, L>>::TOKEN,
        expr = <Op as CombineExpr<L>>::EXPR,
        entry = AXIS_SCAN_ENTRY,
    )
}

impl AxisScanDialect for CudaC {
    #[inline]
    fn axis_scan_source<Op, T>(width: BlockWidth) -> String
    where
        Op: CombineExpr<Self>,
        T: IdentityToken<Op, Self>,
    {
        c_family_axis_scan_source::<Self, Op, T>(width)
    }
}

impl AxisScanDialect for HipC {
    #[inline]
    fn axis_scan_source<Op, T>(width: BlockWidth) -> String
    where
        Op: CombineExpr<Self>,
        T: IdentityToken<Op, Self>,
    {
        c_family_axis_scan_source::<Self, Op, T>(width)
    }
}
