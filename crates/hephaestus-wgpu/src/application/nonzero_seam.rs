//! WGPU implementation of [`hephaestus_core::NonzeroOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK):
//! one thread per coordinates row rescans the input's logical cells in
//! row-major order, latches its row's flat index, and writes that
//! flat's decoded coordinates — the output-parallel scan that keeps row
//! order deterministic without atomics or scratch storage, the scatter
//! seam's determinism precedent. Thread 0 additionally writes the
//! total count, so even a zero-cell input reports through the single
//! launch (the dispatch is at least one thread). The zero test is
//! scalar inequality against zero, never a bitwise test. WGSL does not
//! advertise IEEE special values, so a `NaN` input cell counts as zero
//! on this backend — the shared special-values clause skips `NaN`-mixed
//! inputs here wholesale by construction (ADR 0043) — while `-0.0`
//! counts as zero exactly as on the host.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, NonzeroOps, Result, StridedView,
    Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `NonzeroMeta` uniform: rank-8 padded input shape, input and
/// coordinates strides, `[in_offset, coords_offset, count_offset, _]`,
/// and `[dispatch, numel, ndim, _]`.
///
/// The coordinates are always rank 2, so their padded strides hold the
/// row stride in lane 6 and the column stride in lane 7. The count is a
/// single cell, so only its offset crosses — its stride is irrelevant.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NonzeroMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    coords_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<NonzeroMeta>() == 128);

const WGSL_NONZERO_META: &str = r"struct NonzeroMeta {
    in_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    coords_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn nonzero_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> nmeta: NonzeroMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> coords: array<u32>;
@group(0) @binding(3) var<storage, read_write> count: array<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let r = gid.x;
    let numel = nmeta.params.y;
    if (r > 0u && r >= numel) {{
        return;
    }}
    let ndim = nmeta.params.z;
    var seen = 0u;
    var wanted = numel;
    for (var flat: u32 = 0u; flat < numel; flat = flat + 1u) {{
        var rem = flat;
        var in_off = i32(nmeta.offsets.x);
        for (var p: i32 = 7; p >= 0; p = p - 1) {{
            let group = p / 4;
            let lane_c = p % 4;
            let d = nmeta.in_shape[group][lane_c];
            let c = rem % d;
            rem = rem / d;
            in_off = in_off + i32(c) * nmeta.in_strides[group][lane_c];
        }}
        if (input[u32(in_off)] != {ty}(0)) {{
            if (seen == r) {{
                wanted = flat;
            }}
            seen = seen + 1u;
        }}
    }}
    if (r == 0u) {{
        count[nmeta.offsets.z] = seen;
    }}
    if (wanted < numel) {{
        let first_lane = 8u - ndim;
        var rem = wanted;
        for (var p: i32 = 7; p >= 0; p = p - 1) {{
            let group = p / 4;
            let lane_c = p % 4;
            let d = nmeta.in_shape[group][lane_c];
            let c = rem % d;
            rem = rem / d;
            if (u32(p) >= first_lane) {{
                let col = u32(p) - first_lane;
                let cell = u32(i32(nmeta.offsets.y) + i32(r) * nmeta.coords_strides[1][2] + i32(col) * nmeta.coords_strides[1][3]);
                coords[cell] = c;
            }}
        }}
    }}
}}
"#,
        meta = WGSL_NONZERO_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct NonzeroKernel;

/// WGPU device marker implementing [`hephaestus_core::NonzeroOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuNonzeroOps;

impl<T> NonzeroOps<WgpuDevice, T> for WgpuNonzeroOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;

    fn nonzero_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        coords: StridedView<'_, WgpuBuffer<u32>, 2>,
        count: StridedView<'_, WgpuBuffer<u32>, 1>,
    ) -> Result<()> {
        nonzero_into::<T, N>(device, input, coords, count, BlockWidth::DEFAULT)
    }
}

/// Write nonzero coordinates and their count at a caller-selected
/// [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when the coordinates shape is not
/// `[numel, N]`, the count shape is not `[1]`, any two of the three
/// buffers alias, a layout is unsupported, or the backend dispatch
/// fails.
pub fn nonzero_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    coords: StridedView<'_, WgpuBuffer<u32>, 2>,
    count: StridedView<'_, WgpuBuffer<u32>, 1>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "nonzero dispatch supports rank <= 8");
    }

    hephaestus_core::validate_nonzero_shape(
        input.layout.shape(),
        coords.layout.shape(),
        count.layout.shape(),
    )?;

    let in_layout = input.layout;
    let coords_layout = coords.layout;
    let count_layout = count.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    coords_layout
        .validate_storage_len(coords.buffer.len)
        .map_err(map_layout_err)?;
    count_layout
        .validate_storage_len(count.buffer.len)
        .map_err(map_layout_err)?;
    if !coords_layout.is_injective().map_err(map_layout_err)?
        || !count_layout.is_injective().map_err(map_layout_err)?
    {
        return Err(HephaestusError::DispatchFailed {
            message: "nonzero output layouts must be non-overlapping".to_string(),
        });
    }
    if coords.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "nonzero coords buffer must not alias the input buffer".to_string(),
        });
    }
    if count.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "nonzero count buffer must not alias the input buffer".to_string(),
        });
    }
    if coords.buffer.aliases(count.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "nonzero coords buffer must not alias the count buffer".to_string(),
        });
    }

    // Validated above: the coordinates rows are the input numel, so the
    // dispatch derives from them rather than re-deriving the product.
    // At least one thread launches so thread 0 reports the count of an
    // empty input.
    let numel = coords_layout.shape()[0];
    let dispatch = numel.max(1);

    // An empty storage buffer cannot bind — wgpu requires at least one
    // element per binding — so an empty input or coordinates buffer
    // substitutes a one-cell zeroed dummy for binding. Either is empty
    // only when numel or N is zero, in which case the kernel provably
    // touches neither binding: no scan iteration reads the input, and
    // no thread latches a row (numel zero) or writes a coordinate
    // column (N zero). The count cell always binds its own storage:
    // its `[1]` shape plus the storage-length check above guarantee a
    // nonempty buffer.
    let dummy_input;
    let input_binding: &WgpuBuffer<T> = if input.buffer.len == 0 {
        dummy_input = device.alloc_zeroed::<T>(1)?;
        &dummy_input
    } else {
        input.buffer
    };
    let dummy_coords;
    let coords_binding: &WgpuBuffer<u32> = if coords.buffer.len == 0 {
        dummy_coords = device.alloc_zeroed::<u32>(1)?;
        &dummy_coords
    } else {
        coords.buffer
    };
    let meta = NonzeroMeta {
        in_shape: pad_shape(in_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        coords_strides: pad_strides(coords_layout.strides())?,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(coords_layout.offset(), "coords offset")?,
            to_u32(count_layout.offset(), "count offset")?,
            0,
        ],
        params: [
            to_u32(dispatch, "dispatch size")?,
            to_u32(numel, "scan size")?,
            to_u32(N, "input rank")?,
            0,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<NonzeroMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<NonzeroKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-nonzero",
        || nonzero_shader::<T>(block_width),
    );

    let groups = workgroups(dispatch, block_width)?;

    let mut entries = BindGroupEntries::with_capacity(4);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: meta_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: input_binding.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 2,
        resource: coords_binding.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: count.buffer.buffer.as_entire_binding(),
    });
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-nonzero"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-nonzero"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-nonzero",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
