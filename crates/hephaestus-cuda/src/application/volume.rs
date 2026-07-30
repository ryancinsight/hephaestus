//! CUDA volume ray-integral dispatch.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DispatchGrid, MultiStorageKernel, Result,
    validate_ray_line_integrals,
};

use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

pub use hephaestus_core::{FieldGeometry, RAY_STRIDE};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VolumeParams {
    values: [f32; 11],
}

const VOLUME_KERNEL: &str = r#"
struct VolumeParams {
    float values[11];
};

__device__ float sample_trilinear(
    const float* field,
    float3 p,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz
) {
    float3 maxn = make_float3((float)(nx - 1u), (float)(ny - 1u), (float)(nz - 1u));
    if (p.x < 0.0f || p.y < 0.0f || p.z < 0.0f ||
        p.x > maxn.x || p.y > maxn.y || p.z > maxn.z) {
        return 0.0f;
    }
    float3 f = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    unsigned int lx = (unsigned int)f.x;
    unsigned int ly = (unsigned int)f.y;
    unsigned int lz = (unsigned int)f.z;
    unsigned int hx = min(lx + 1u, nx - 1u);
    unsigned int hy = min(ly + 1u, ny - 1u);
    unsigned int hz = min(lz + 1u, nz - 1u);
    float3 t = make_float3(p.x - f.x, p.y - f.y, p.z - f.z);
    unsigned int i000 = (lx * ny + ly) * nz + lz;
    unsigned int i100 = (hx * ny + ly) * nz + lz;
    unsigned int i010 = (lx * ny + hy) * nz + lz;
    unsigned int i110 = (hx * ny + hy) * nz + lz;
    unsigned int i001 = (lx * ny + ly) * nz + hz;
    unsigned int i101 = (hx * ny + ly) * nz + hz;
    unsigned int i011 = (lx * ny + hy) * nz + hz;
    unsigned int i111 = (hx * ny + hy) * nz + hz;
    float c00 = field[i000] + (field[i100] - field[i000]) * t.x;
    float c10 = field[i010] + (field[i110] - field[i010]) * t.x;
    float c01 = field[i001] + (field[i101] - field[i001]) * t.x;
    float c11 = field[i011] + (field[i111] - field[i011]) * t.x;
    float c0 = c00 + (c10 - c00) * t.y;
    float c1 = c01 + (c11 - c01) * t.y;
    return c0 + (c1 - c0) * t.z;
}

extern "C" __global__ void ray_line_integrals(
    const float* field,
    const float* rays,
    float* out,
    VolumeParams params
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n_rays = (unsigned int)params.values[3];
    if (i >= n_rays) {
        return;
    }
    unsigned int nx = (unsigned int)params.values[0];
    unsigned int ny = (unsigned int)params.values[1];
    unsigned int nz = (unsigned int)params.values[2];
    float3 origin = make_float3(params.values[4], params.values[5], params.values[6]);
    float3 spacing = make_float3(params.values[7], params.values[8], params.values[9]);
    float step = params.values[10];
    float3 o = make_float3(rays[i * 6u], rays[i * 6u + 1u], rays[i * 6u + 2u]);
    float3 d = make_float3(rays[i * 6u + 3u], rays[i * 6u + 4u], rays[i * 6u + 5u]);
    float3 bmin = origin;
    float3 bmax = make_float3(
        origin.x + (float)(nx - 1u) * spacing.x,
        origin.y + (float)(ny - 1u) * spacing.y,
        origin.z + (float)(nz - 1u) * spacing.z
    );
    float3 inv = make_float3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
    float3 t1 = make_float3(
        (bmin.x - o.x) * inv.x,
        (bmin.y - o.y) * inv.y,
        (bmin.z - o.z) * inv.z
    );
    float3 t2 = make_float3(
        (bmax.x - o.x) * inv.x,
        (bmax.y - o.y) * inv.y,
        (bmax.z - o.z) * inv.z
    );
    float3 tmin = make_float3(fminf(t1.x, t2.x), fminf(t1.y, t2.y), fminf(t1.z, t2.z));
    float3 tmax = make_float3(fmaxf(t1.x, t2.x), fmaxf(t1.y, t2.y), fmaxf(t1.z, t2.z));
    float t_enter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float t_exit = fminf(fminf(tmax.x, tmax.y), tmax.z);
    if (t_exit < t_enter) {
        out[i] = 0.0f;
        return;
    }
    float len = t_exit - t_enter;
    if (len <= 0.0f) {
        out[i] = 0.0f;
        return;
    }
    unsigned int n = max((unsigned int)ceilf(len / step), 1u);
    float actual = len / (float)n;
    float acc = 0.0f;
    for (unsigned int s = 0u; s < n; ++s) {
        float tmid = t_enter + ((float)s + 0.5f) * actual;
        float3 p = make_float3(
            (o.x + d.x * tmid - origin.x) / spacing.x,
            (o.y + d.y * tmid - origin.y) / spacing.y,
            (o.z + d.z * tmid - origin.z) / spacing.z
        );
        acc += sample_trilinear(field, p, nx, ny, nz);
    }
    out[i] = acc * actual;
}
"#;

/// Integrate a device-resident field along packed device-resident rays.
pub fn ray_line_integrals_into(
    device: &CudaDevice,
    field: &CudaBuffer<f32>,
    geometry: FieldGeometry,
    rays: &CudaBuffer<f32>,
    step: f32,
    out: &CudaBuffer<f32>,
    width: BlockWidth,
) -> Result<()> {
    let n_rays = validate_ray_line_integrals(field.len(), geometry, rays.len(), out.len(), step)?;
    if n_rays == 0 {
        return Ok(());
    }
    let width_usize = usize::try_from(width.get()).map_err(|_| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("volume block width {} exceeds usize range", width.get()),
        }
    })?;
    let groups = n_rays.div_ceil(width_usize);
    let grid = DispatchGrid::new(
        u32::try_from(groups).map_err(|_| hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("volume ray grid {groups} exceeds u32 range"),
        })?,
        1,
        1,
    );
    let params = VolumeParams {
        values: [
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
        ],
    };
    let kernel = CudaMultiStorageKernel::new(
        "hephaestus-volume-ray-integral",
        VOLUME_KERNEL,
        "ray_line_integrals",
        &[0, 1, 2],
        [width.get(), 1, 1],
        0,
    )?;
    MultiStorageKernel::<CudaDevice, VolumeParams, [CudaStorageBinding<'_>; 3]>::dispatch(
        &kernel,
        device,
        [
            CudaStorageBinding::new(0, field),
            CudaStorageBinding::new(1, rays),
            CudaStorageBinding::new(2, out),
        ],
        &params,
        grid,
    )
}

/// Allocate output storage and integrate a device-resident field along rays.
pub fn ray_line_integrals(
    device: &CudaDevice,
    field: &CudaBuffer<f32>,
    geometry: FieldGeometry,
    rays: &CudaBuffer<f32>,
    n_rays: usize,
    step: f32,
    width: BlockWidth,
) -> Result<CudaBuffer<f32>> {
    validate_ray_line_integrals(field.len(), geometry, rays.len(), n_rays, step)?;
    let out = device.alloc_uninitialized::<f32>(n_rays)?;
    ray_line_integrals_into(device, field, geometry, rays, step, &out, width)?;
    Ok(out)
}
