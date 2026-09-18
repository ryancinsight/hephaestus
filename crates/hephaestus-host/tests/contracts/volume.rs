//! Host instantiation of the shared ray-integral clause, plus the
//! host-specific cases it does not reach.

use hephaestus_conformance::assert_ray_integral_contract;
use hephaestus_core::{ComputeDevice, FieldGeometry, HephaestusError, RayIntegralOps};
use hephaestus_host::{HostDevice, HostRayIntegralOps};

#[test]
fn host_satisfies_the_ray_integral_contract() {
    assert_ray_integral_contract(&HostDevice::new(), &HostRayIntegralOps);
}

const GEOMETRY: FieldGeometry = FieldGeometry {
    dims: [3, 3, 3],
    origin: [0.0, 0.0, 0.0],
    spacing: [1.0, 1.0, 1.0],
};

/// A diagonal ray through a uniform field integrates to the value times the
/// chord `2·sqrt(3)`. The 14 midpoint samples of 0.5 sum exactly, so the
/// error is the chord's own rounding (the unit direction, the slab products,
/// the segment division) and the final multiply: a few ULPs, bounded by 8ε.
#[test]
fn a_diagonal_chord_through_a_uniform_field_integrates_to_its_length() {
    let device = HostDevice::new();
    let field = device.upload(&[0.5f32; 27]).expect("field");
    let unit = 1.0 / 3.0f32.sqrt();
    let rays = device
        .upload(&[-1.0f32, -1.0, -1.0, unit, unit, unit])
        .expect("rays");
    let out = device.alloc_zeroed::<f32>(1).expect("out");
    HostRayIntegralOps
        .ray_line_integrals_into(&device, &field, GEOMETRY, &rays, 0.25, &out)
        .expect("integrate");
    let mut got = [0.0f32; 1];
    device.download(&out, &mut got).expect("download");
    let expected = 0.5 * 2.0 * 3.0f32.sqrt();
    assert!(
        (got[0] - expected).abs() <= 8.0 * f32::EPSILON * expected,
        "{} != {expected}",
        got[0]
    );
}

#[test]
fn an_output_aliasing_the_field_is_rejected() {
    let device = HostDevice::new();
    let field = device.upload(&[0.5f32; 27]).expect("field");
    let rays = device
        .upload(&[-1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0])
        .expect("rays");
    let error = HostRayIntegralOps
        .ray_line_integrals_into(&device, &field, GEOMETRY, &rays, 0.5, &field.clone())
        .expect_err("aliased output must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{error:?}"
    );
}
