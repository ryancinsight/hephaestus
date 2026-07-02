//! Volume ray integrals: trilinearly-sampled line integrals of a 3-D scalar
//! field along a batch of rays.
//!
//! The computed-tomography / dose ray-trace primitive: for each ray, clip to the
//! field's node-centre bounding box, march `n = ceil(L/step)` midpoint samples of
//! the trilinearly-interpolated field, and return `Σ field(mid_s) · (L/n)` — the
//! discrete `∫ field dl` in world units. One GPU thread per ray, so a full
//! sinogram/beamlet batch dispatches in a single kernel with the field resident
//! on the device (upload the volume once, download one scalar per ray).
//!
//! Semantics match a midpoint ray-marcher exactly: rays that miss the box (or
//! graze with `L ≤ 0`) integrate to `0`; samples falling outside the node range
//! after clipping (floating-point boundary rounding) contribute `0`.

use bytemuck::Pod;

use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

/// Number of `f32` lanes per packed ray (`[ox, oy, oz, dx, dy, dz]`).
pub const RAY_STRIDE: usize = 6;

/// World-space description of the sampled field: C-contiguous `(x, y, z)` order
/// with `z` fastest (`flat = (ix·ny + iy)·nz + iz`), node `i` at
/// `origin + i·spacing` per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldGeometry {
    /// Node counts per axis `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// World coordinate of node `(0, 0, 0)` per axis.
    pub origin: [f32; 3],
    /// Node pitch per axis (world units, > 0).
    pub spacing: [f32; 3],
}

fn shader_source(width: BlockWidth) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<storage, read> rays: array<f32>;
// [nx, ny, nz, n_rays, origin xyz, spacing xyz, step] — counts are exact f32s
// (validated < 2^24 host-side), keeping the kernel within the 4-storage-buffer
// per-stage device limit.
@group(0) @binding(2) var<storage, read> pf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn sample_trilinear(p: vec3<f32>, dims: vec3<u32>) -> f32 {{
    let maxn = vec3<f32>(dims - vec3<u32>(1u));
    if (any(p < vec3<f32>(0.0)) || any(p > maxn)) {{
        return 0.0;
    }}
    let f = floor(p);
    let lo = vec3<u32>(f);
    let hi = min(lo + vec3<u32>(1u), dims - vec3<u32>(1u));
    let t = p - f;
    let ny = dims.y;
    let nz = dims.z;
    let v000 = field[(lo.x * ny + lo.y) * nz + lo.z];
    let v100 = field[(hi.x * ny + lo.y) * nz + lo.z];
    let v010 = field[(lo.x * ny + hi.y) * nz + lo.z];
    let v110 = field[(hi.x * ny + hi.y) * nz + lo.z];
    let v001 = field[(lo.x * ny + lo.y) * nz + hi.z];
    let v101 = field[(hi.x * ny + lo.y) * nz + hi.z];
    let v011 = field[(lo.x * ny + hi.y) * nz + hi.z];
    let v111 = field[(hi.x * ny + hi.y) * nz + hi.z];
    let c00 = mix(v000, v100, t.x);
    let c10 = mix(v010, v110, t.x);
    let c01 = mix(v001, v101, t.x);
    let c11 = mix(v011, v111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z);
}}

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n_rays = u32(pf[3]);
    if (i >= n_rays) {{
        return;
    }}
    let o = vec3<f32>(rays[i * 6u], rays[i * 6u + 1u], rays[i * 6u + 2u]);
    let d = vec3<f32>(rays[i * 6u + 3u], rays[i * 6u + 4u], rays[i * 6u + 5u]);
    let dims = vec3<u32>(u32(pf[0]), u32(pf[1]), u32(pf[2]));
    let origin = vec3<f32>(pf[4], pf[5], pf[6]);
    let spacing = vec3<f32>(pf[7], pf[8], pf[9]);
    let step = pf[10];
    let bmin = origin;
    let bmax = origin + vec3<f32>(dims - vec3<u32>(1u)) * spacing;

    // Slab intersection with the node-centre AABB (axis-parallel rays handled by
    // the inf products resolving to +/-inf, min/max discarding them).
    let inv = vec3<f32>(1.0) / d;
    let t1 = (bmin - o) * inv;
    let t2 = (bmax - o) * inv;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    var t_enter = max(max(tmin.x, tmin.y), tmin.z);
    var t_exit = min(min(tmax.x, tmax.y), tmax.z);
    if (t_exit < t_enter) {{
        out[i] = 0.0;
        return;
    }}
    let len = t_exit - t_enter;
    if (len <= 0.0) {{
        out[i] = 0.0;
        return;
    }}
    let n = max(u32(ceil(len / step)), 1u);
    let actual = len / f32(n);
    var acc = 0.0;
    for (var s = 0u; s < n; s = s + 1u) {{
        let tmid = t_enter + (f32(s) + 0.5) * actual;
        let p = (o + d * tmid - origin) / spacing;
        acc = acc + sample_trilinear(p, dims);
    }}
    out[i] = acc * actual;
}}
"#,
        wg = width.get(),
    )
}

/// Integrate `field` along each packed ray, writing one integral per ray to `out`.
///
/// `rays` holds [`RAY_STRIDE`] `f32`s per ray (`origin.xyz`, then a direction
/// whose length defines the parameterization — pass unit directions so `step`
/// and the result are in world units). `out.len` selects the ray count; misses
/// integrate to `0`.
///
/// # Errors
/// [`HephaestusError`] if `rays` is not `out.len × RAY_STRIDE` long, the field
/// length does not match `geometry.dims`, or dispatch fails.
pub fn ray_line_integrals_into(
    device: &WgpuDevice,
    field: &WgpuBuffer<f32>,
    geometry: FieldGeometry,
    rays: &WgpuBuffer<f32>,
    step: f32,
    out: &WgpuBuffer<f32>,
    width: BlockWidth,
) -> Result<()> {
    let n_rays = out.len;
    if rays.len != n_rays * RAY_STRIDE {
        return Err(HephaestusError::LengthMismatch {
            host_len: rays.len,
            device_len: n_rays * RAY_STRIDE,
        });
    }
    let n_field = geometry.dims.iter().map(|&d| d as usize).product::<usize>();
    if field.len != n_field {
        return Err(HephaestusError::LengthMismatch {
            host_len: field.len,
            device_len: n_field,
        });
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("ray-march step must be finite and positive, got {step}"),
        });
    }
    if n_rays == 0 {
        return Ok(());
    }

    // Counts travel as f32 lanes (single params buffer, staying within the
    // 4-storage-buffer per-stage device limit); exactness requires < 2^24.
    const F32_EXACT_LIMIT: usize = 1 << 24;
    for (label, count) in [
        ("dims.x", geometry.dims[0] as usize),
        ("dims.y", geometry.dims[1] as usize),
        ("dims.z", geometry.dims[2] as usize),
        ("n_rays", n_rays),
    ] {
        if count >= F32_EXACT_LIMIT {
            return Err(HephaestusError::DispatchFailed {
                message: format!("{label} = {count} exceeds the exact-f32 limit 2^24"),
            });
        }
    }
    let pf = device.upload(&[
        geometry.dims[0] as f32,
        geometry.dims[1] as f32,
        geometry.dims[2] as f32,
        n_rays as f32,
        geometry.origin[0],
        geometry.origin[1],
        geometry.origin[2],
        geometry.spacing[0],
        geometry.spacing[1],
        geometry.spacing[2],
        step,
    ])?;

    let groups = workgroups(n_rays, width)?;
    let key = (
        std::any::TypeId::of::<RayIntegralKernel>(),
        std::any::TypeId::of::<f32>(),
        width.get(),
    );
    let pipeline = cached_pipeline(device, key, "hephaestus-volume-ray-integral", || {
        shader_source(width)
    });

    encode_ray_integral(
        device,
        &pipeline,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: field.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rays.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pf.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.as_entire_binding(),
            },
        ],
        groups,
    )
}

/// Pipeline-cache key marker for the ray-integral kernel.
#[derive(Clone, Copy, Debug)]
struct RayIntegralKernel;

fn encode_ray_integral(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry<'_>],
    groups: u32,
) -> Result<()> {
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-volume-ray-integral"),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-volume-ray-integral"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-volume-ray-integral"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

/// Convenience wrapper allocating the output buffer.
///
/// # Errors
/// As [`ray_line_integrals_into`].
pub fn ray_line_integrals(
    device: &WgpuDevice,
    field: &WgpuBuffer<f32>,
    geometry: FieldGeometry,
    rays: &WgpuBuffer<f32>,
    n_rays: usize,
    step: f32,
    width: BlockWidth,
) -> Result<WgpuBuffer<f32>>
where
    f32: Pod,
{
    let out = device.alloc_zeroed::<f32>(n_rays)?;
    ray_line_integrals_into(device, field, geometry, rays, step, &out, width)?;
    Ok(out)
}
