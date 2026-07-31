//! Backend-neutral contracts for volume ray-integral dispatch.

use crate::{HephaestusError, Result};

/// Number of `f32` lanes per packed ray (`origin.xyz`, then `direction.xyz`).
pub const RAY_STRIDE: usize = 6;

/// World-space description of a C-contiguous `(x, y, z)` sampled field.
///
/// The flattened element index is `(ix * ny + iy) * nz + iz`; node `i` is at
/// `origin + i * spacing` on each axis. Spacing and origin are validated by
/// [`validate_ray_line_integrals`] before a backend launches a kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldGeometry {
    /// Node counts per axis `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// World coordinate of node `(0, 0, 0)` per axis.
    pub origin: [f32; 3],
    /// Positive node pitch per axis in world units.
    pub spacing: [f32; 3],
}

/// Validate the shared ray-integral storage and parameter contract.
///
/// The returned ray count is the logical output length. The backend must keep
/// the field and packed rays device-resident and write one scalar per ray.
/// Counts are limited to the exactly representable `f32` range because the
/// backend parameter ABIs carry dimensions and ray count in `f32` lanes.
pub fn validate_ray_line_integrals(
    field_len: usize,
    geometry: FieldGeometry,
    rays_len: usize,
    output_len: usize,
    step: f32,
) -> Result<usize> {
    let n_rays = output_len;
    if geometry.dims.contains(&0) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "ray field dimensions must be positive, got {:?}",
                geometry.dims
            ),
        });
    }
    let expected_rays =
        n_rays
            .checked_mul(RAY_STRIDE)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("ray storage length overflows for {n_rays} rays"),
            })?;
    if rays_len != expected_rays {
        return Err(HephaestusError::LengthMismatch {
            host_len: rays_len,
            device_len: expected_rays,
        });
    }

    let expected_field = geometry
        .dims
        .into_iter()
        .try_fold(1usize, |product, dimension| {
            let dimension =
                usize::try_from(dimension).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("field dimension {dimension} does not fit host indexing"),
                })?;
            product
                .checked_mul(dimension)
                .ok_or_else(|| HephaestusError::DispatchFailed {
                    message: format!(
                        "field dimensions {:?} overflow host indexing",
                        geometry.dims
                    ),
                })
        })?;
    if field_len != expected_field {
        return Err(HephaestusError::LengthMismatch {
            host_len: field_len,
            device_len: expected_field,
        });
    }

    if !(step.is_finite() && step > 0.0) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("ray-march step must be finite and positive, got {step}"),
        });
    }
    if geometry
        .origin
        .into_iter()
        .chain(geometry.spacing)
        .any(|value| !value.is_finite())
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: "ray field origin and spacing must be finite".to_string(),
        });
    }
    if geometry.spacing.iter().any(|&value| value <= 0.0) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "ray field spacing must be positive, got {:?}",
                geometry.spacing
            ),
        });
    }

    if n_rays == 0 {
        return Ok(0);
    }

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
    Ok(n_rays)
}

use crate::domain::device::ComputeDevice;

/// Device-neutral volume ray line integrals.
///
/// Marches each ray through a regular `f32` scalar field with midpoint
/// sampling and accumulates `field · dl`. The scalar type is fixed at `f32`
/// by every backend kernel; a generic scalar dimension enters here when the
/// kernels ship it, not before. Ray records are `RAY_STRIDE` floats:
/// `[origin xyz, direction xyz]`.
///
/// Implementors are zero-sized per-backend markers. A bound of
/// `R: RayIntegralOps<D>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
pub trait RayIntegralOps<D: ComputeDevice> {
    /// Integrate `field` along each ray into caller-owned `out` (one value
    /// per ray), stepping by `step` world units.
    ///
    /// # Errors
    ///
    /// Returns a geometry/ray/output length mismatch, a non-positive step,
    /// or the backend dispatch failure.
    fn ray_line_integrals_into(
        &self,
        device: &D,
        field: &D::Buffer<f32>,
        geometry: FieldGeometry,
        rays: &D::Buffer<f32>,
        step: f32,
        out: &D::Buffer<f32>,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEOMETRY: FieldGeometry = FieldGeometry {
        dims: [2, 3, 4],
        origin: [0.0, 1.0, -2.0],
        spacing: [0.5, 1.0, 2.0],
    };

    #[test]
    fn valid_contract_returns_logical_ray_count() {
        assert_eq!(
            validate_ray_line_integrals(24, GEOMETRY, 12, 2, 0.25).expect("valid volume"),
            2
        );
    }

    #[test]
    fn empty_output_still_requires_exact_storage_shapes() {
        assert_eq!(
            validate_ray_line_integrals(24, GEOMETRY, 0, 0, 0.25).expect("empty volume"),
            0
        );
        assert!(matches!(
            validate_ray_line_integrals(24, GEOMETRY, 1, 0, 0.25),
            Err(HephaestusError::LengthMismatch {
                host_len: 1,
                device_len: 0
            })
        ));
    }

    #[test]
    fn invalid_step_spacing_and_dimensions_are_typed_configuration_errors() {
        for step in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                validate_ray_line_integrals(24, GEOMETRY, 6, 1, step),
                Err(HephaestusError::DispatchFailed { .. })
            ));
        }
        let mut bad_spacing = GEOMETRY;
        bad_spacing.spacing[1] = 0.0;
        assert!(matches!(
            validate_ray_line_integrals(24, bad_spacing, 6, 1, 0.25),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
        let mut bad_dimensions = GEOMETRY;
        bad_dimensions.dims[2] = 0;
        assert!(matches!(
            validate_ray_line_integrals(0, bad_dimensions, 6, 1, 0.25),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn exact_f32_count_boundary_is_rejected() {
        let too_many = 1 << 24;
        let result =
            validate_ray_line_integrals(24, GEOMETRY, too_many * RAY_STRIDE, too_many, 0.25);
        assert!(matches!(
            result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("n_rays") && message.contains("2^24")
        ));
    }
}
