//! Strided-layout-aware binary dispatch over leto host-side layout metadata.
//!
//! Consumers describe operands with [`leto::Layout`] (shape/strides/offset) so
//! transposed, sliced, and broadcast (zero-stride) views dispatch directly —
//! no host-side materialization into contiguous staging copies. Inputs
//! broadcast to the output shape with leto's own broadcast rules, keeping
//! device semantics identical to leto's CPU `binary_map`.

use core::marker::PhantomData;
use std::any::TypeId;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result};
use leto::Layout;
use wgpu::util::DeviceExt;

use crate::application::elementwise::BinaryWgslOp;
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP_SIZE: u32 = 256;

/// Maximum rank the packed rank-4 metadata covers. Lower-rank layouts are
/// padded with leading size-1 / stride-0 dimensions, which contribute nothing
/// to the offset computation.
pub const MAX_STRIDED_RANK: usize = 4;

/// Pipeline-cache discriminator so strided kernels never collide with the
/// contiguous kernels of the same `Op` in the `(TypeId, TypeId)` cache key.
struct StridedBinaryKernel<Op>(PhantomData<Op>);

/// Packed layout metadata matching the WGSL `Meta` uniform: rank-4 padded
/// shape, per-operand strides, and `[a_off, b_off, out_off, len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StridedMeta {
    shape: [u32; 4],
    a_strides: [i32; 4],
    b_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
}

#[inline]
fn pad_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 4]> {
    let mut out = [1u32; 4];
    for (d, &dim) in shape.iter().enumerate() {
        out[4 - N + d] = u32::try_from(dim).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("dimension {dim} exceeds u32 range"),
        })?;
    }
    Ok(out)
}

#[inline]
fn pad_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 4]> {
    let mut out = [0i32; 4];
    for (d, &stride) in strides.iter().enumerate() {
        out[4 - N + d] = i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("stride {stride} exceeds i32 range"),
        })?;
    }
    Ok(out)
}

fn shader_source<Op: BinaryWgslOp, T: WgslScalar>() -> String {
    format!(
        r#"struct Meta {{
    shape: vec4<u32>,
    a_strides: vec4<i32>,
    b_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
}}

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
    var rem = i;
    var a_off = i32(lmeta.offsets.x);
    var b_off = i32(lmeta.offsets.y);
    var o_off = i32(lmeta.offsets.z);
    for (var d: i32 = 3; d >= 0; d = d - 1) {{
        let dim = lmeta.shape[d];
        let idx = i32(rem % dim);
        rem = rem / dim;
        a_off = a_off + idx * lmeta.a_strides[d];
        b_off = b_off + idx * lmeta.b_strides[d];
        o_off = o_off + idx * lmeta.out_strides[d];
    }}
    let lhs = a[u32(a_off)];
    let rhs = b[u32(b_off)];
    out[u32(o_off)] = {expr};
}}
"#,
        ty = T::WGSL_TYPE,
        wg = WORKGROUP_SIZE,
        expr = Op::WGSL_EXPR,
    )
}

/// Run `out[idx] = op(a[idx], b[idx])` over logical indices of `out_layout`,
/// with `a`/`b` broadcast to the output shape by leto's broadcast rules.
///
/// All three operands are described by leto host-side layouts; transposed,
/// sliced, offset, and zero-stride (broadcast) inputs dispatch without any
/// contiguous materialization. The output buffer is caller-owned (allocation
/// control stays with the consumer); zero-stride aliasing in the output
/// layout is rejected because concurrent invocations would race on one
/// physical element. Rank is capped at [`MAX_STRIDED_RANK`] at compile time.
pub fn binary_elementwise_strided_into<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    a_layout: &Layout<N>,
    b: &WgpuBuffer<T>,
    b_layout: &Layout<N>,
    out: &WgpuBuffer<T>,
    out_layout: &Layout<N>,
) -> Result<()>
where
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let map_layout_err = |e: leto::LetoError| HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    };

    // Broadcast inputs to the output shape (leto semantics; zero strides on
    // expanded singleton axes — pure metadata, no data movement).
    let a_layout = a_layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    let b_layout = b_layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;

    if out_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    a_layout
        .validate_storage_len(a.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(out.len)
        .map_err(map_layout_err)?;

    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(());
    }

    let to_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("{what} {value} exceeds u32 range"),
        })
    };
    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        b_strides: pad_strides(b_layout.strides)?,
        out_strides: pad_strides(out_layout.strides)?,
        offsets: [
            to_u32(a_layout.offset, "input offset")?,
            to_u32(b_layout.offset, "input offset")?,
            to_u32(out_layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };
    let meta_buffer = device
        .inner()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hephaestus-strided-meta"),
            contents: bytemuck::bytes_of(&meta),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let key = (TypeId::of::<StridedBinaryKernel<Op>>(), TypeId::of::<T>());
    let pipeline = {
        let mut cache = device.pipeline_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            cached.clone()
        } else {
            let module = device
                .inner()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("hephaestus-strided"),
                    source: wgpu::ShaderSource::Wgsl(shader_source::<Op, T>().into()),
                });
            let pipeline =
                device
                    .inner()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("hephaestus-strided"),
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });
            cache.insert(key, pipeline.clone());
            pipeline
        }
    };

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-strided"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-strided"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = to_u32(len.div_ceil(WORKGROUP_SIZE as usize), "workgroup count")?;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}
