//! Volume ray line integrals on the host reference device (ADR 0046).
//!
//! [`HostRayIntegralOps`] evaluates the seam's midpoint ray-march in `f32`,
//! the same arithmetic the device kernels perform: each ray is clipped to the
//! node-centre bounding box by the slab method, the chord is split into
//! `max(ceil(len / step), 1)` equal segments, and each segment's midpoint
//! sample of the trilinearly interpolated field is weighted by the segment
//! length. Interpolation is leto-ops'
//! [`trilinear_index_space`](leto_ops::trilinear_index_space); a midpoint
//! outside the node range contributes zero, as in the kernels, where leto's
//! interpolation would clamp.

use hephaestus_core::{
    FieldGeometry, HephaestusError, RAY_STRIDE, RayIntegralOps, Result, validate_ray_line_integrals,
};
use leto::{ArrayView, Layout};
use leto_ops::trilinear_index_space;

use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Volume ray line integrals for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostRayIntegralOps;

/// Parameter `t` along a ray where it enters and leaves the node-centre box,
/// or `None` when it misses or grazes it.
fn chord(origin: [f32; 3], direction: [f32; 3], geometry: &FieldGeometry) -> Option<(f32, f32)> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let low = geometry.origin[axis];
        // `dims - 1` nodes span the box; dims are validated below 2^24, so
        // the conversion is exact.
        let high = low + (geometry.dims[axis] - 1) as f32 * geometry.spacing[axis];
        // An axis-parallel ray divides by zero; the resulting infinities fall
        // out of the min/max below exactly as in the kernels.
        let inverse = 1.0 / direction[axis];
        let near = (low - origin[axis]) * inverse;
        let far = (high - origin[axis]) * inverse;
        enter = enter.max(near.min(far));
        exit = exit.min(near.max(far));
    }
    (exit - enter > 0.0).then_some((enter, exit))
}

/// Number of march segments for a chord, as the kernels count them.
fn segment_count(length: f32, step: f32) -> Result<u32> {
    let segments = (length / step).ceil().max(1.0);
    // `u32::MAX as f32` rounds up to 2^32, so the bound is strict: every
    // whole `f32` below it converts exactly.
    if segments < u32::MAX as f32 {
        Ok(segments as u32)
    } else {
        Err(HephaestusError::DispatchFailed {
            message: format!(
                "a ray chord of length {length} at step {step} needs more than u32::MAX segments"
            ),
        })
    }
}

impl RayIntegralOps<HostDevice> for HostRayIntegralOps {
    fn ray_line_integrals_into(
        &self,
        _device: &HostDevice,
        field: &HostBuffer<f32>,
        geometry: FieldGeometry,
        rays: &HostBuffer<f32>,
        step: f32,
        out: &HostBuffer<f32>,
    ) -> Result<()> {
        require_disjoint_output(field, rays, out)?;
        let out_len = out.read().len();
        let field_cells = field.read();
        let ray_cells = rays.read();
        validate_ray_line_integrals(field_cells.len(), geometry, ray_cells.len(), out_len, step)?;
        let dims = geometry.dims.map(|extent| extent as usize);
        let layout = Layout::c_contiguous(dims).map_err(map_leto_error)?;
        let volume = ArrayView::try_new(layout, &field_cells).map_err(map_leto_error)?;
        let last_node = geometry.dims.map(|extent| (extent - 1) as f32);

        let mut integrals = Vec::with_capacity(out_len);
        for ray in ray_cells.chunks_exact(RAY_STRIDE) {
            let origin = [ray[0], ray[1], ray[2]];
            let direction = [ray[3], ray[4], ray[5]];
            let Some((enter, exit)) = chord(origin, direction, &geometry) else {
                integrals.push(0.0);
                continue;
            };
            let length = exit - enter;
            let segments = segment_count(length, step)?;
            let segment = length / segments as f32;
            let mut sum = 0.0f32;
            for index in 0..segments {
                let t = enter + (index as f32 + 0.5) * segment;
                let node = [0, 1, 2].map(|axis| {
                    (origin[axis] + direction[axis] * t - geometry.origin[axis])
                        / geometry.spacing[axis]
                });
                let inside = node
                    .iter()
                    .zip(last_node)
                    .all(|(&coordinate, last)| (0.0..=last).contains(&coordinate));
                if inside {
                    sum += trilinear_index_space(volume, node[0], node[1], node[2]);
                }
            }
            integrals.push(sum * segment);
        }
        out.write().copy_from_slice(&integrals);
        Ok(())
    }
}
