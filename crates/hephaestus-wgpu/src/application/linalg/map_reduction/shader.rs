//! WGSL source generation for fused map-reduction first passes.

use hephaestus_core::{BlockWidth, CombineExpr, DialectScalar, IdentityToken, Wgsl};

use super::MapReductionOp;
use crate::application::strided::{WGSL_DECODE, WGSL_META};

pub(super) fn source<Op, T>(width: BlockWidth) -> String
where
    Op: MapReductionOp,
    T: DialectScalar<Wgsl> + IdentityToken<Op::ReduceOp, Wgsl>,
{
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let i = global_id.x;
    if (i < lmeta.offsets.w) {{
{decode}        let lhs = a[u32(a_off)];
        let rhs = b[u32(b_off)];
        shared_data[local_id.x] = {map_expr};
    }} else {{
        shared_data[local_id.x] = {identity};
    }}

    workgroupBarrier();

    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (local_id.x < stride) {{
            let lhs = shared_data[local_id.x];
            let rhs = shared_data[local_id.x + stride];
            shared_data[local_id.x] = {reduce_expr};
        }}
        workgroupBarrier();
    }}

    if (local_id.x == 0u) {{
        out[workgroup_id.x] = shared_data[0];
    }}
}}
"#,
        meta = WGSL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = WGSL_DECODE,
        identity = <T as IdentityToken<Op::ReduceOp, Wgsl>>::TOKEN,
        map_expr = Op::WGSL_MAP_EXPR,
        reduce_expr = <Op::ReduceOp as CombineExpr<Wgsl>>::EXPR,
    )
}
