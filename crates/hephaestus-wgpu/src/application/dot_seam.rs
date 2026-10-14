//! WGPU implementation of [`hephaestus_core::DotOps`].
//!
//! One packed-metadata kernel serves every rank pair up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK):
//! a single thread folds the elementwise products in ascending flat
//! order, decoding each side's logical coordinates with the padded
//! rank-8 odometer. One thread is the whole parallel strategy, and
//! that is deliberate: any tree or tiling would reassociate the sum
//! and break the bitwise float agreement the seam pins, so the kernel
//! is exactly the host's single-pass fold, running on the device. A
//! blocked parallel fold with a pinned association order is a
//! follow-up; it must prove bitwise equality against this fold before
//! it replaces it. The kernel is pure arithmetic with no value
//! comparison, so non-finite propagation follows the scalar
//! arithmetic — there is no dialect branch.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, DotOps, HephaestusError, Result, StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `DotMeta` uniform: rank-8 padded shapes and strides of both
/// inputs, `[a_offset, b_offset, out_offset, _]`, and
/// `[numel, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DotMeta {
    a_shape: [u32; 8],
    a_strides: [i32; 8],
    b_shape: [u32; 8],
    b_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<DotMeta>() == 160);

const WGSL_DOT_META: &str = r"struct DotMeta {
    a_shape: array<vec4<u32>, 2>,
    a_strides: array<vec4<i32>, 2>,
    b_shape: array<vec4<u32>, 2>,
    b_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn dot_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> dmeta: DotMeta;
@group(0) @binding(1) var<storage, read> a_in: array<{ty}>;
@group(0) @binding(2) var<storage, read> b_in: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{
        return;
    }}
    let numel = dmeta.params.x;
    var acc = {ty}(0);
    for (var flat: u32 = 0u; flat < numel; flat = flat + 1u) {{
        var rem = flat;
        var a_off = i32(dmeta.offsets.x);
        for (var p: i32 = 7; p >= 0; p = p - 1) {{
            let group = p / 4;
            let lane_c = p % 4;
            let d = dmeta.a_shape[group][lane_c];
            let c = rem % d;
            rem = rem / d;
            a_off = a_off + i32(c) * dmeta.a_strides[group][lane_c];
        }}
        rem = flat;
        var b_off = i32(dmeta.offsets.y);
        for (var p: i32 = 7; p >= 0; p = p - 1) {{
            let group = p / 4;
            let lane_c = p % 4;
            let d = dmeta.b_shape[group][lane_c];
            let c = rem % d;
            rem = rem / d;
            b_off = b_off + i32(c) * dmeta.b_strides[group][lane_c];
        }}
        acc = acc + a_in[u32(a_off)] * b_in[u32(b_off)];
    }}
    out[dmeta.offsets.z] = acc;
}}
"#,
        meta = WGSL_DOT_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct DotKernel;

/// WGPU device marker implementing [`hephaestus_core::DotOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuDotOps;

impl<T> DotOps<WgpuDevice, T> for WgpuDotOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn dot_into<const N: usize, const M: usize>(
        &self,
        device: &WgpuDevice,
        a: StridedView<'_, WgpuBuffer<T>, N>,
        b: StridedView<'_, WgpuBuffer<T>, M>,
        out: StridedView<'_, WgpuBuffer<T>, 1>,
    ) -> Result<()> {
        dot_into::<T, N, M>(device, a, b, out, BlockWidth::DEFAULT)
    }
}

/// Fold the inner product at a caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when the input numels disagree or
/// overflow, the output shape is not `[1]`, the output buffer aliases
/// either input buffer, a layout is unsupported, or the backend
/// dispatch fails.
pub fn dot_into<T, const N: usize, const M: usize>(
    device: &WgpuDevice,
    a: StridedView<'_, WgpuBuffer<T>, N>,
    b: StridedView<'_, WgpuBuffer<T>, M>,
    out: StridedView<'_, WgpuBuffer<T>, 1>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "dot dispatch supports rank <= 8");
        assert!(M <= MAX_STRIDED_RANK, "dot dispatch supports rank <= 8");
    }

    hephaestus_core::validate_dot_shape(a.layout.shape(), b.layout.shape(), out.layout.shape())?;

    let a_layout = a.layout;
    let b_layout = b.layout;
    let out_layout = out.layout;
    a_layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(out.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "dot output layout must be non-overlapping".to_string(),
        });
    }
    if out.buffer.aliases(a.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "dot output buffer must not alias the first input buffer".to_string(),
        });
    }
    if out.buffer.aliases(b.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "dot output buffer must not alias the second input buffer".to_string(),
        });
    }

    // Validated above: both numels agree, so the fold bound derives
    // from `a` rather than re-deriving either product.
    let numel: usize = a_layout.shape().iter().product();
    let meta = DotMeta {
        a_shape: pad_shape(a_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_shape: pad_shape(b_layout.shape())?,
        b_strides: pad_strides(b_layout.strides())?,
        offsets: [
            to_u32(a_layout.offset(), "a offset")?,
            to_u32(b_layout.offset(), "b offset")?,
            to_u32(out_layout.offset(), "out offset")?,
            0,
        ],
        params: [to_u32(numel, "fold size")?, 0, 0, 0],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<DotMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    // An empty storage buffer cannot bind — wgpu requires at least one
    // element per binding — so an empty input substitutes a one-cell
    // zeroed dummy for binding. Either input is empty only when numel
    // is zero, in which case the fold reads neither binding. The
    // output cell always binds its own storage: its `[1]` shape plus
    // the storage-length check above guarantee a nonempty buffer.
    let dummy_a;
    let a_binding: &WgpuBuffer<T> = if a.buffer.len == 0 {
        dummy_a = device.alloc_zeroed::<T>(1)?;
        &dummy_a
    } else {
        a.buffer
    };
    let dummy_b;
    let b_binding: &WgpuBuffer<T> = if b.buffer.len == 0 {
        dummy_b = device.alloc_zeroed::<T>(1)?;
        &dummy_b
    } else {
        b.buffer
    };

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<DotKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-dot",
        || dot_shader::<T>(block_width),
    );

    let groups = workgroups(1, block_width)?;

    let mut entries = BindGroupEntries::with_capacity(4);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: meta_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: a_binding.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 2,
        resource: b_binding.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: out.buffer.buffer.as_entire_binding(),
    });
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-dot"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-dot"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-dot",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
