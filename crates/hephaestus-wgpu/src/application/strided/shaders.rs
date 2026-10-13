use hephaestus_core::{BinaryExpr, BlockWidth, DialectScalar, UnaryExpr, Wgsl};

/// WGSL `Meta` declaration shared by every strided kernel.
pub(crate) const WGSL_META: &str = r"struct Meta {
    shape: array<vec4<u32>, 2>,
    a_strides: array<vec4<i32>, 2>,
    b_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

/// Flat-index → per-operand-offset decode shared by every strided kernel.
/// `arrayLength` cannot guard strided access, so the logical length travels
/// in `offsets.w` and is checked by each kernel before this body runs.
pub(crate) const WGSL_DECODE: &str = r"    var rem = i;
    var a_off = i32(lmeta.offsets.x);
    var b_off = i32(lmeta.offsets.y);
    var o_off = i32(lmeta.offsets.z);
    for (var d: i32 = 7; d >= 0; d = d - 1) {
        let group = d / 4;
        let lane = d % 4;
        let dim = lmeta.shape[group][lane];
        let idx = i32(rem % dim);
        rem = rem / dim;
        a_off = a_off + idx * lmeta.a_strides[group][lane];
        b_off = b_off + idx * lmeta.b_strides[group][lane];
        o_off = o_off + idx * lmeta.out_strides[group][lane];
    }
";

pub(crate) fn binary_shader<T: DialectScalar<Wgsl>>(
    width: BlockWidth,
    expr: &'static str,
) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{
        return;
    }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = b[u32(b_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = expr,
    )
}

pub(crate) fn unary_shader<Op: UnaryExpr<Wgsl>, T: DialectScalar<Wgsl>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{
        return;
    }}
{decode}    let x = a[u32(a_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = <Op as UnaryExpr<Wgsl>>::EXPR,
    )
}

/// Strided scalar kernel: one strided storage operand `a`, the broadcast scalar
/// supplied through a small `uniform` (no per-call storage operand), strided
/// output. Mirrors [`scalar_elementwise_into`](crate::application::elementwise::scalar) but
/// over leto strided layouts. The shared `WGSL_DECODE` computes an unused
/// `b_off` here (identical to the unary kernel), which the WGSL compiler elides.
pub(crate) fn scalar_shader<Op: BinaryExpr<Wgsl>, T: DialectScalar<Wgsl>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<uniform> scalar: {ty};
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{
        return;
    }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = scalar;
    out[u32(o_off)] = {expr};
}}
"#,
        meta = WGSL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = WGSL_DECODE,
        expr = <Op as BinaryExpr<Wgsl>>::EXPR,
    )
}
