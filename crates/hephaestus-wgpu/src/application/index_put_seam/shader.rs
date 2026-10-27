//! WGSL uniform metadata structs and shader builders for `index_put`.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, DialectScalar, Wgsl};

/// WGSL `IndexPutNdMeta` uniform: rank-8 padded output and values shapes,
/// index, values, and output strides, `[index_offset, values_offset,
/// output_offset, _]`, and `[dispatch_len, sel_len, sel_lane, dim_extent]`.
///
/// `sel_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes. The
/// rank-1 index stride packs into the trailing lane (`7`). The input needs
/// no entry: the copy kernel runs first and the update seeds from the
/// copied output cell.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct IndexPutNdMeta {
    pub(super) out_shape: [u32; 8],
    pub(super) vals_shape: [u32; 8],
    pub(super) idx_strides: [i32; 8],
    pub(super) vals_strides: [i32; 8],
    pub(super) out_strides: [i32; 8],
    pub(super) offsets: [u32; 4],
    pub(super) params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexPutNdMeta>() == 192);

const WGSL_INDEX_PUT_ND_META: &str = r"struct IndexPutNdMeta {
    out_shape: array<vec4<u32>, 2>,
    vals_shape: array<vec4<u32>, 2>,
    idx_strides: array<vec4<i32>, 2>,
    vals_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

pub(super) fn index_put_replace_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: IndexPutNdMeta;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.params.x) {{
        return;
    }}
    let sel_lane = i32(smeta.params.z);
    var rem = i;
    var vals_off = i32(smeta.offsets.y);
    var out_off = i32(smeta.offsets.z);
    var slot = 0u;
    var dim_coord = 0u;
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let vals_dim = smeta.vals_shape[group][lane];
        let c = rem % vals_dim;
        rem = rem / vals_dim;
        vals_off = vals_off + i32(c) * smeta.vals_strides[group][lane];
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
        if (p == sel_lane) {{
            slot = c;
            dim_coord = c;
        }}
    }}
    // The rank-1 index stride lives in the trailing padded lane; the
    // leading lanes pad zeros.
    let idx_off = i32(smeta.offsets.x) + i32(slot) * smeta.idx_strides[1][3];
    let g = indices[u32(idx_off)];
    if (g >= smeta.params.w) {{
        atomicStore(&error_flag, 1u);
        return;
    }}
    let dim_stride = smeta.out_strides[sel_lane / 4][sel_lane % 4];
    out_off = out_off - i32(dim_coord) * dim_stride + i32(g) * dim_stride;
    out[u32(out_off)] = values[u32(vals_off)];
}}
"#,
        meta = WGSL_INDEX_PUT_ND_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

pub(super) fn index_put_accumulate_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: IndexPutNdMeta;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.params.x) {{
        return;
    }}
    let sel_lane = i32(smeta.params.z);
    let dim_extent = smeta.params.w;
    let dim_stride = smeta.out_strides[sel_lane / 4][sel_lane % 4];
    let vals_dim_stride = smeta.vals_strides[sel_lane / 4][sel_lane % 4];
    var rem = i;
    var out_off = i32(smeta.offsets.z);
    // The values offset with the `dim` lane zeroed: each scan step adds
    // back `s * vals_dim_stride`.
    var vals_base = i32(smeta.offsets.y);
    var wanted = 0u;
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = smeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
        if (p != sel_lane) {{
            vals_base = vals_base + i32(c) * smeta.vals_strides[group][lane];
        }} else {{
            wanted = c;
        }}
    }}
    // Seeded from the copied output cell: the copy kernel ran first, and
    // this thread owns its cell, so the read shares with no other thread.
    var acc: {ty} = out[u32(out_off)];
    let sel_len = smeta.params.y;
    let idx_base = i32(smeta.offsets.x);
    let idx_stride = smeta.idx_strides[1][3];
    for (var s: u32 = 0u; s < sel_len; s = s + 1u) {{
        let g = indices[u32(idx_base + i32(s) * idx_stride)];
        if (g >= dim_extent) {{
            atomicStore(&error_flag, 1u);
            continue;
        }}
        if (g == wanted) {{
            acc = acc + values[u32(vals_base + i32(s) * vals_dim_stride)];
        }}
    }}
    out[u32(out_off)] = acc;
}}
"#,
        meta = WGSL_INDEX_PUT_ND_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGSL `IndexPutCopyMeta` uniform: rank-8 padded output shape, input and
/// output strides, and `[input_offset, output_offset, len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct IndexPutCopyMeta {
    pub(super) out_shape: [u32; 8],
    pub(super) in_strides: [i32; 8],
    pub(super) out_strides: [i32; 8],
    pub(super) offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexPutCopyMeta>() == 112);

const WGSL_INDEX_PUT_COPY_META: &str = r"struct IndexPutCopyMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

pub(super) fn index_put_copy_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: IndexPutCopyMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= cmeta.offsets.z) {{
        return;
    }}
    var rem = i;
    var in_off = i32(cmeta.offsets.x);
    var out_off = i32(cmeta.offsets.y);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = cmeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + i32(c) * cmeta.in_strides[group][lane];
        out_off = out_off + i32(c) * cmeta.out_strides[group][lane];
    }}
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_INDEX_PUT_COPY_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}
