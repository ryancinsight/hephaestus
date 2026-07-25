//! Metal volume ray-integral delegation.

use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, Result, validate_ray_line_integrals,
};
use hephaestus_wgpu as wgpu_backend;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

pub use hephaestus_core::{FieldGeometry, RAY_STRIDE};

/// Integrate a device-resident field along packed device-resident rays through
/// the native Metal-selected WGPU device.
pub fn ray_line_integrals_into(
    device: &MetalDevice,
    field: &MetalBuffer<f32>,
    geometry: FieldGeometry,
    rays: &MetalBuffer<f32>,
    step: f32,
    out: &MetalBuffer<f32>,
    width: BlockWidth,
) -> Result<()> {
    wgpu_backend::ray_line_integrals_into(
        device.wgpu_device(),
        field.wgpu_buffer(),
        geometry,
        rays.wgpu_buffer(),
        step,
        out.wgpu_buffer(),
        width,
    )
}

/// Allocate output storage and integrate a device-resident field along rays.
pub fn ray_line_integrals(
    device: &MetalDevice,
    field: &MetalBuffer<f32>,
    geometry: FieldGeometry,
    rays: &MetalBuffer<f32>,
    n_rays: usize,
    step: f32,
    width: BlockWidth,
) -> Result<MetalBuffer<f32>> {
    validate_ray_line_integrals(field.len(), geometry, rays.len(), n_rays, step)?;
    let out = device.alloc_zeroed::<f32>(n_rays)?;
    ray_line_integrals_into(device, field, geometry, rays, step, &out, width)?;
    Ok(out)
}
