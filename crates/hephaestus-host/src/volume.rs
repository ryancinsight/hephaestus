//! Volume ray line integrals on the host reference device (ADR 0046).
//!
//! [`HostRayIntegralOps`] evaluates the seam's midpoint ray-march in `f32`,
//! the same arithmetic the device kernels perform: each ray is clipped to the
//! node-centre bounding box by the slab method and to the forward half-line
//! `t >= 0`, the parameter chord is scaled by the direction length into world
//! units, the world chord is split into `max(ceil(len / step), 1)` equal
//! segments, and each segment's midpoint sample of the trilinearly
//! interpolated field is weighted by the segment length. Interpolation is leto-ops'
//! [`trilinear_index_space`](leto_ops::trilinear_index_space); a midpoint
//! outside the node range contributes zero, as in the kernels, where leto's
//! interpolation would clamp.

use eunomia::FloatElement;
use eunomia::convert::IntegerTarget;
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
/// or `None` when it misses, grazes, or points away from it.
///
/// The line chord is clamped to the forward half-line: the domain is the ray
/// from its origin, so an inside origin starts at `t = 0` and a backward ray
/// misses. The caller rejects a zero direction before calling: every axis
/// would divide by zero and an origin on a box face would form `0 * inf`.
fn chord(origin: [f32; 3], direction: [f32; 3], geometry: &FieldGeometry) -> Option<(f32, f32)> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let low = geometry.origin[axis];
        // `dims - 1` nodes span the box; dims are validated below 2^24, so
        // the conversion is exact (single round-to-nearest, as `as` performs).
        let high =
            low + f32::from_integer(i64::from(geometry.dims[axis] - 1)) * geometry.spacing[axis];
        // An axis-parallel ray divides by zero; the resulting infinities fall
        // out of the min/max below exactly as in the kernels.
        let inverse = 1.0 / direction[axis];
        let near = (low - origin[axis]) * inverse;
        let far = (high - origin[axis]) * inverse;
        enter = enter.max(near.min(far));
        exit = exit.min(near.max(far));
    }
    let enter = enter.max(0.0);
    (exit - enter > 0.0).then_some((enter, exit))
}

/// Number of march segments for a chord, as the kernels count them.
fn segment_count(length: f32, step: f32) -> Result<u32> {
    let segments = (length / step).ceil().max(1.0);
    // `u32::MAX as f32` rounds up to 2^32, spelled here as a literal so no
    // conversion appears: the bound is strict, and every whole `f32` below
    // it converts exactly.
    if segments < 4_294_967_296.0 {
        u32::try_from_rounded(segments).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("ray segment count {segments} is not a whole u32"),
        })
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
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(geometry.dims) {
            *slot = usize::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("volume extent {extent} exceeds usize"),
            })?;
        }
        let layout = Layout::c_contiguous(dims).map_err(map_leto_error)?;
        let volume = ArrayView::try_new(layout, &field_cells).map_err(map_leto_error)?;
        let last_node = geometry
            .dims
            .map(|extent| f32::from_integer(i64::from(extent - 1)));

        let mut integrals = Vec::with_capacity(out_len);
        for ray in ray_cells.chunks_exact(RAY_STRIDE) {
            let origin = [ray[0], ray[1], ray[2]];
            let direction = [ray[3], ray[4], ray[5]];
            // A zero direction carries no world direction; see `chord`.
            if direction.iter().all(|&component| component == 0.0) {
                integrals.push(0.0);
                continue;
            }
            let Some((enter, exit)) = chord(origin, direction, &geometry) else {
                integrals.push(0.0);
                continue;
            };
            // The march length is in world units: scale the parameter chord
            // by the direction length so non-unit directions integrate
            // `field · dl`, as in the kernels. The midpoint parameter still
            // advances by the parameter segment while each sample weighs the
            // world segment, so the march covers exactly the clamped chord.
            let direction_length = direction
                .iter()
                .map(|&component| component * component)
                .sum::<f32>()
                .sqrt();
            let parameter_length = exit - enter;
            let length = parameter_length * direction_length;
            let segments = segment_count(length, step)?;
            let segment = length / f32::from_integer(i64::from(segments));
            let parameter_step = parameter_length / f32::from_integer(i64::from(segments));
            let mut sum = 0.0f32;
            for index in 0..segments {
                let t = enter + (f32::from_integer(i64::from(index)) + 0.5) * parameter_step;
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
