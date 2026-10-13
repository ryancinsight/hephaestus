//! Householder reflector trailing updates.

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{HephaestusError, Result, TrailingHh};
use std::any::TypeId;

/// Packed metadata for the panel Householder reflector application kernel.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct HhMeta {
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    matrix_cols: u32,
    k: u32,
}

/// Per-reflector metadata consumed by the panel Householder kernel.
///
/// Public because the WGPU [`BlockedQrBackend`] impl names it as its
/// `Reflectors` buffer element, and a public trait impl cannot leak a
/// restricted type.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub struct HhReflectorMeta {
    /// Offset of this reflector in the packed vector buffer.
    pub(super) vector_offset: u32,
    /// Householder scale factor β.
    pub(super) beta: f32,
}

// ---------------------------------------------------------------------------
// Householder apply kernel:  A[:, col] -= β · v · (vᵀ · A[:, col])
// ---------------------------------------------------------------------------

/// WGSL source for applying all panel Householder reflectors.
fn hh_shader_source() -> String {
    r#"struct HhMeta {
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    matrix_cols: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read>      v_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> a_buf: array<f32>;

struct ReflectorMeta {
    vector_offset: u32,
    beta: f32,
}
@group(0) @binding(2) var<storage, read>      reflector_buf: array<ReflectorMeta>;
@group(0) @binding(3) var<uniform>             params: HhMeta;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)

fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wid:  vec3<u32>,
) {
    let col = params.k + params.reflector_count + wid.x;
    let tid = lid.x;

    if (col >= params.matrix_cols) {
        return;
    }

    let n_cols = params.matrix_cols;
    let k_offset = params.k;

    for (var reflector = 0u; reflector < params.reflector_count; reflector = reflector + 1u) {
        let n_rows = params.panel_rows - reflector;
        let start_row = k_offset + reflector;
        let v_off = reflector_buf[reflector].vector_offset;
        let beta = reflector_buf[reflector].beta;

        // Phase 1: partial dot = vᵀ · A[start_row:m, col]
        var partial = f32(0.0);
        var row = tid;
        while (row < n_rows) {
            let a_idx = (start_row + row) * n_cols + col;
            partial = partial + v_buf[v_off + row] * a_buf[a_idx];
            row = row + 256u;
        }
        sdata[tid] = partial;
        workgroupBarrier();

        // Parallel tree reduction.
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (tid < s) {
                sdata[tid] = sdata[tid] + sdata[tid + s];
            }
            workgroupBarrier();
        }

        let dot = sdata[0];
        workgroupBarrier();

        // Phase 2: A[start_row:m, col] -= beta * v * dot
        row = tid;
        while (row < n_rows) {
            let a_idx = (start_row + row) * n_cols + col;
            a_buf[a_idx] = a_buf[a_idx] - beta * v_buf[v_off + row] * dot;
            row = row + 256u;
        }
        storageBarrier();
        workgroupBarrier();
    }
}
"#
    .to_string()
}

struct HhKernel;

/// Apply the packed panel Householder reflectors to a matrix's trailing columns.
///
/// The shader pipeline is cached; the bind group and the uniform buffer are
/// built per call, because the trait method that reaches this takes no
/// loop-scoped descriptors.
pub(crate) fn householder_trailing_update(
    device: &WgpuDevice,
    vectors: &WgpuBuffer<f32>,
    matrix: &WgpuBuffer<f32>,
    reflectors: &WgpuBuffer<HhReflectorMeta>,
    spec: TrailingHh<'_>,
) -> Result<()> {
    fn dimension(value: usize, name: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("{name} {value} exceeds u32"),
        })
    }

    let reflector_host: Vec<HhReflectorMeta> = spec
        .vector_offsets
        .iter()
        .copied()
        .zip(spec.betas.iter().copied())
        .map(|(offset, beta)| -> Result<HhReflectorMeta> {
            Ok(HhReflectorMeta {
                vector_offset: dimension(offset, "HH vector offset")?,
                beta,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    device.write_sub_buffer(reflectors, 0, &reflector_host)?;

    let hh_meta = HhMeta {
        panel_rows: dimension(spec.panel_rows, "HH panel_rows")?,
        reflector_count: dimension(spec.betas.len(), "HH reflector_count")?,
        trail_cols: dimension(spec.trail_cols, "HH trail_cols")?,
        matrix_cols: dimension(spec.matrix_cols, "HH matrix_cols")?,
        k: dimension(spec.panel_start, "HH panel start")?,
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<HhKernel>(), TypeId::of::<f32>(), 256),
        "hephaestus-hh",
        hh_shader_source,
    );
    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<HhMeta>(1)?)?;
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-hh-panel"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vectors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reflectors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });
    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&hh_meta));

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-qr-hh-update"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-hh-panel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dimension(spec.trail_cols, "HH workgroup count")?, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

// ---------------------------------------------------------------------------
// Q accumulation uniform
// ---------------------------------------------------------------------------
