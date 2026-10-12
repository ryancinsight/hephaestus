use std::marker::PhantomData;

use hephaestus_core::{BlockWidth, CombineExpr, IdentityToken, SumOp, Wgsl};

/// ZST wrapper to generate a unique TypeId in the pipeline cache for reduction operations.
pub(super) struct ReductionOpWrapper<Op>(PhantomData<Op>);
pub(super) struct ReductionFinalOpWrapper<Op>(PhantomData<Op>);
pub(super) struct AxisReductionParallelKernel<Op>(PhantomData<Op>);
pub(super) struct AxisReductionAxis0TiledKernel<Op>(PhantomData<Op>);
pub(super) struct MeanAxisParallelKernel<T>(PhantomData<T>);
pub(super) struct MeanAxis0TiledKernel<T>(PhantomData<T>);

const AXIS0_TILE_COLUMNS: u32 = 64;
pub(super) fn shader_source<Op: CombineExpr<Wgsl>, T: IdentityToken<Op, Wgsl>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"{prelude}@group(0) @binding(0) var<storage, read> input: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let i = global_id.x;
    let num_elements = arrayLength(&input);
    
    if (i < num_elements) {{
        shared_data[local_id.x] = input[i];
    }} else {{
        shared_data[local_id.x] = {identity};
    }}
    
    workgroupBarrier();
    
    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (local_id.x < stride) {{
            let lhs = shared_data[local_id.x];
            let rhs = shared_data[local_id.x + stride];
            shared_data[local_id.x] = {expr};
        }}
        workgroupBarrier();
    }}
    
    if (local_id.x == 0u) {{
        output[workgroup_id.x] = shared_data[0];
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

pub(super) fn final_reduction_shader_source<Op: CombineExpr<Wgsl>, T: IdentityToken<Op, Wgsl>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"{prelude}@group(0) @binding(0) var<storage, read> input: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {{
    let lane = local_id.x;
    let num_elements = arrayLength(&input);
    var acc: {ty} = {identity};
    var i = lane;
    loop {{
        if (i >= num_elements) {{
            break;
        }}
        let lhs = acc;
        let rhs = input[i];
        acc = {expr};
        i = i + {wg}u;
    }}
    shared_data[lane] = acc;
    workgroupBarrier();

    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (lane < stride) {{
            let lhs = shared_data[lane];
            let rhs = shared_data[lane + stride];
            shared_data[lane] = {expr};
        }}
        workgroupBarrier();
    }}

    if (lane == 0u) {{
        output[0u] = shared_data[0];
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

/// Grid-strided workgroup-parallel axis reduction (WG-P5): each of the `wg`
/// lanes accumulates every `axis_len`-th element starting at its lane index
/// (a per-lane sequential fold, not a single load), then the lanes'
/// `wg` partials tree-reduce as before. Correct and fully lane-parallel for
/// any `axis_len` — including `axis_len > wg`, where the previous "parallel"
/// kernel could only read the first `wg` elements (each lane loaded at most
/// one) and dispatch fell back to a genuinely serial one-thread-per-row
/// kernel that did zero cross-lane work. This one kernel now covers both
/// regimes; the serial kernel is dead and removed.
///
/// The per-lane stride-accumulation reassociates the combine relative to a
/// purely sequential fold (elements interleave across lanes before the tree
/// reduction), so exact bitwise equality with a sequential CPU reference is
/// no longer guaranteed for inputs where the combine is not exact under
/// reassociation (e.g. float sums with real rounding). Differential tests
/// against a derived epsilon bound this (`axis_reduction_grid_stride_...`
/// contract tests); small-integer-valued fixtures remain exact because
/// integer-valued f32 sums have no rounding error under any grouping.
pub(super) fn axis_reduction_parallel_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: CombineExpr<Wgsl>,
    T: IdentityToken<Op, Wgsl>,
{
    format!(
        r#"{prelude}
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let i = workgroup_id.x;
    let lane = local_id.x;
    if (i >= axis_meta.offsets.w) {{
        return;
    }}

    let axis = axis_meta.offsets.z;
    let axis_len = select(axis_meta.input_shape.y, axis_meta.input_shape.x, axis == 0u);
    let out_row = select(i, 0u, axis == 0u);
    let out_col = select(0u, i, axis == 0u);

    var acc: {ty} = {identity};
    var idx = lane;
    loop {{
        if (idx >= axis_len) {{
            break;
        }}
        let in_row = select(out_row, idx, axis == 0u);
        let in_col = select(idx, out_col, axis == 0u);
        let in_off = i32(axis_meta.offsets.x)
            + i32(in_row) * axis_meta.input_strides.x
            + i32(in_col) * axis_meta.input_strides.y;
        let lhs = acc;
        let rhs = input[u32(in_off)];
        acc = {expr};
        idx = idx + {wg}u;
    }}
    partials[lane] = acc;
    workgroupBarrier();

    var stride = {wg}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (lane < stride) {{
            let lhs = partials[lane];
            let rhs = partials[lane + stride];
            partials[lane] = {expr};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (lane == 0u) {{
        let out_off = i32(axis_meta.offsets.y)
            + i32(out_row) * axis_meta.output_strides.x
            + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[0u];
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

/// Grid-strided workgroup-parallel mean-axis reduction (WG-P5); see
/// [`axis_reduction_parallel_shader_source`] for the rationale (same fix,
/// applied to the sum-then-divide mean kernel).
pub(super) fn mean_axis_parallel_shader_source<T>(width: BlockWidth) -> String
where
    T: IdentityToken<SumOp, Wgsl>,
{
    format!(
        r#"{prelude}
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let i = workgroup_id.x;
    let lane = local_id.x;
    if (i >= axis_meta.offsets.w) {{
        return;
    }}

    let axis = axis_meta.offsets.z;
    let axis_len = select(axis_meta.input_shape.y, axis_meta.input_shape.x, axis == 0u);
    let out_row = select(i, 0u, axis == 0u);
    let out_col = select(0u, i, axis == 0u);

    var acc: {ty} = {identity};
    var idx = lane;
    loop {{
        if (idx >= axis_len) {{
            break;
        }}
        let in_row = select(out_row, idx, axis == 0u);
        let in_col = select(idx, out_col, axis == 0u);
        let in_off = i32(axis_meta.offsets.x)
            + i32(in_row) * axis_meta.input_strides.x
            + i32(in_col) * axis_meta.input_strides.y;
        acc = acc + input[u32(in_off)];
        idx = idx + {wg}u;
    }}
    partials[lane] = acc;
    workgroupBarrier();

    var stride = {wg}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (lane < stride) {{
            partials[lane] = partials[lane] + partials[lane + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (lane == 0u) {{
        let out_off = i32(axis_meta.offsets.y)
            + i32(out_row) * axis_meta.output_strides.x
            + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[0u] / {ty}(axis_len);
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<SumOp, Wgsl>>::TOKEN,
    )
}

pub(super) fn axis0_tile_shape(width: BlockWidth) -> (u32, u32) {
    let tile_cols = AXIS0_TILE_COLUMNS.min(width.get());
    (tile_cols, width.get() / tile_cols)
}

pub(super) fn axis_reduction_axis0_tiled_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: CombineExpr<Wgsl>,
    T: IdentityToken<Op, Wgsl>,
{
    let (tile_cols, tile_rows) = axis0_tile_shape(width);
    format!(
        r#"{prelude}
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({tile_cols}, {tile_rows})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let out_col = workgroup_id.x * {tile_cols}u + local_id.x;
    let row_lane = local_id.y;
    let partial_idx = row_lane * {tile_cols}u + local_id.x;
    let axis_len = axis_meta.input_shape.x;

    var value: {ty} = {identity};
    if (out_col < axis_meta.offsets.w) {{
        var row = row_lane;
        loop {{
            if (row >= axis_len) {{
                break;
            }}
            let in_off = i32(axis_meta.offsets.x)
                + i32(row) * axis_meta.input_strides.x
                + i32(out_col) * axis_meta.input_strides.y;
            let lhs = value;
            let rhs = input[u32(in_off)];
            value = {expr};
            row = row + {tile_rows}u;
        }}
    }}
    partials[partial_idx] = value;
    workgroupBarrier();

    var stride = {tile_rows}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (row_lane < stride) {{
            let lhs = partials[partial_idx];
            let rhs = partials[(row_lane + stride) * {tile_cols}u + local_id.x];
            partials[partial_idx] = {expr};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (row_lane == 0u && out_col < axis_meta.offsets.w) {{
        let out_off = i32(axis_meta.offsets.y) + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[local_id.x];
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        tile_cols = tile_cols,
        tile_rows = tile_rows,
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

pub(super) fn mean_axis0_tiled_shader_source<T>(width: BlockWidth) -> String
where
    T: IdentityToken<SumOp, Wgsl>,
{
    let (tile_cols, tile_rows) = axis0_tile_shape(width);
    format!(
        r#"{prelude}
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({tile_cols}, {tile_rows})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let out_col = workgroup_id.x * {tile_cols}u + local_id.x;
    let row_lane = local_id.y;
    let partial_idx = row_lane * {tile_cols}u + local_id.x;
    let axis_len = axis_meta.input_shape.x;

    var value: {ty} = {identity};
    if (out_col < axis_meta.offsets.w) {{
        var row = row_lane;
        loop {{
            if (row >= axis_len) {{
                break;
            }}
            let in_off = i32(axis_meta.offsets.x)
                + i32(row) * axis_meta.input_strides.x
                + i32(out_col) * axis_meta.input_strides.y;
            value = value + input[u32(in_off)];
            row = row + {tile_rows}u;
        }}
    }}
    partials[partial_idx] = value;
    workgroupBarrier();

    var stride = {tile_rows}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (row_lane < stride) {{
            partials[partial_idx] = partials[partial_idx]
                + partials[(row_lane + stride) * {tile_cols}u + local_id.x];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (row_lane == 0u && out_col < axis_meta.offsets.w) {{
        let out_off = i32(axis_meta.offsets.y) + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[local_id.x] / {ty}(axis_len);
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        tile_cols = tile_cols,
        tile_rows = tile_rows,
        identity = <T as IdentityToken<SumOp, Wgsl>>::TOKEN,
    )
}
