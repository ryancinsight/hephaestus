//! WGSL uniform metadata structs and shader builders for `scatter_add`.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, DialectScalar, Wgsl};

/// WGSL `ScatterNdMeta` uniform: rank-8 padded output and index shapes,
/// index, source, and output strides, `[index_offset, src_offset,
/// output_offset, _]`, and `[output_len, src_len, dim_lane, dim_extent]`.
///
/// `dim_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes. The
/// input needs no entry: the copy kernel runs first and the scan seeds
/// from the copied output cell.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ScatterNdMeta {
    pub(super) out_shape: [u32; 8],
    pub(super) idx_shape: [u32; 8],
    pub(super) idx_strides: [i32; 8],
    pub(super) src_strides: [i32; 8],
    pub(super) out_strides: [i32; 8],
    pub(super) offsets: [u32; 4],
    pub(super) params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<ScatterNdMeta>() == 192);

const WGSL_SCATTER_ND_META: &str = r"struct ScatterNdMeta {
    out_shape: array<vec4<u32>, 2>,
    idx_shape: array<vec4<u32>, 2>,
    idx_strides: array<vec4<i32>, 2>,
    src_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

pub(super) fn scatter_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: ScatterNdMeta;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.params.x) {{
        return;
    }}
    let dim_lane = i32(smeta.params.z);
    let dim_extent = smeta.params.w;
    var rem = i;
    var out_off = i32(smeta.offsets.z);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = smeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
    }}
    // Seeded from the copied output cell: the copy kernel ran first, and
    // this thread owns its cell, so the read shares with no other thread.
    var acc: {ty} = out[u32(out_off)];
    let src_len = smeta.params.y;
    for (var j: u32 = 0u; j < src_len; j = j + 1u) {{
        var srem = j;
        var idx_off = i32(smeta.offsets.x);
        var src_off = i32(smeta.offsets.y);
        var mapped = i32(smeta.offsets.z);
        for (var p: i32 = 7; p >= 0; p = p - 1) {{
            let group = p / 4;
            let lane = p % 4;
            let idx_dim = smeta.idx_shape[group][lane];
            let c = srem % idx_dim;
            srem = srem / idx_dim;
            idx_off = idx_off + i32(c) * smeta.idx_strides[group][lane];
            src_off = src_off + i32(c) * smeta.src_strides[group][lane];
            if (p != dim_lane) {{
                mapped = mapped + i32(c) * smeta.out_strides[group][lane];
            }}
        }}
        let g = indices[u32(idx_off)];
        if (g >= dim_extent) {{
            atomicStore(&error_flag, 1u);
            continue;
        }}
        mapped = mapped + i32(g) * smeta.out_strides[dim_lane / 4][dim_lane % 4];
        if (mapped == out_off) {{
            acc = acc + src[u32(src_off)];
        }}
    }}
    out[u32(out_off)] = acc;
}}
"#,
        meta = WGSL_SCATTER_ND_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGSL `ScatterCopyMeta` uniform: rank-8 padded output shape, input and
/// output strides, and `[input_offset, output_offset, len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ScatterCopyMeta {
    pub(super) out_shape: [u32; 8],
    pub(super) in_strides: [i32; 8],
    pub(super) out_strides: [i32; 8],
    pub(super) offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<ScatterCopyMeta>() == 112);

const WGSL_SCATTER_COPY_META: &str = r"struct ScatterCopyMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

pub(super) fn scatter_copy_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: ScatterCopyMeta;
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
        meta = WGSL_SCATTER_COPY_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<ScatterNdMeta>(), 192);
        assert_eq!(core::mem::size_of::<ScatterCopyMeta>(), 112);
    }
}
