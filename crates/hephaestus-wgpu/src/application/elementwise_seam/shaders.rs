use hephaestus_core::{BlockWidth, DialectScalar, UnaryExpr, Wgsl};

pub(crate) fn unary_shader<T: DialectScalar<Wgsl>, Op: UnaryExpr<Wgsl>>(
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
    if (i >= lmeta.offsets.w) {{ return; }}
{decode}    let x = a[u32(a_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = crate::application::strided::WGSL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = crate::application::strided::WGSL_DECODE,
        expr = <Op as UnaryExpr<Wgsl>>::EXPR,
    )
}

pub(crate) fn binary_shader<T: DialectScalar<Wgsl>>(width: BlockWidth, expr: &str) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= lmeta.offsets.w) {{ return; }}
{decode}    let lhs = a[u32(a_off)];
    let rhs = b[u32(b_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        meta = crate::application::strided::WGSL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        decode = crate::application::strided::WGSL_DECODE,
        expr = expr,
    )
}
