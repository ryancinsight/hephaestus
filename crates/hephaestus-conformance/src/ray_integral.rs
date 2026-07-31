//! Contract clauses for device-neutral volume ray line integrals.
//!
//! Each clause is generic over the device and the seam, so every backend runs
//! the same assertions against analytical oracles: a uniform field integrates
//! to `value × chord length`, midpoint sampling integrates an affine field
//! exactly, a uniform integral is step-size independent, and a ray missing
//! the volume integrates to zero.
//!
//! ## Tolerance derivation
//!
//! The march accumulates at most `chord / step = 16 / 0.125 = 128` terms of
//! magnitude ≤ `0.5`. Naive `f32` summation error is bounded by
//! `n · ε · Σ|terms| ≈ 128 · 1.2e-7 · 4 ≈ 6e-5`; the assertions use `1e-4`,
//! under twice that bound. The miss oracle is exact: no step samples the
//! volume, so the accumulator never leaves zero.

use hephaestus_core::{ComputeDevice, FieldGeometry, RAY_STRIDE, RayIntegralOps};

/// Absolute bound derived above from the deepest march in these clauses.
const SUM_BOUND: f32 = 1e-4;

/// 9x5x5 field, spacing 2: an 18x10x10 world-unit volume at the origin.
fn geometry() -> FieldGeometry {
    FieldGeometry {
        dims: [9, 5, 5],
        origin: [0.0, 0.0, 0.0],
        spacing: [2.0, 2.0, 2.0],
    }
}

/// Build the row-major host field from an index function.
fn build_field(f: impl Fn(u32, u32, u32) -> f32) -> Vec<f32> {
    let g = geometry();
    let mut host = Vec::new();
    for ix in 0..g.dims[0] {
        for iy in 0..g.dims[1] {
            for iz in 0..g.dims[2] {
                host.push(f(ix, iy, iz));
            }
        }
    }
    host
}

/// Integrate `rays` over `host_field` with `step`, returning one value per ray.
fn run<D, R>(device: &D, ops: &R, host_field: &[f32], rays: &[f32], step: f32) -> Vec<f32>
where
    D: ComputeDevice,
    R: RayIntegralOps<D>,
{
    let field = device.upload(host_field).expect("field upload");
    let ray_buf = device.upload(rays).expect("ray upload");
    let n = rays.len() / RAY_STRIDE;
    let out = device.alloc_zeroed::<f32>(n).expect("output alloc");
    ops.ray_line_integrals_into(device, &field, geometry(), &ray_buf, step, &out)
        .expect("ray integral dispatch");
    let mut got = vec![0.0f32; n];
    device.download(&out, &mut got).expect("download");
    got
}

/// Run every ray-integral clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_ray_integral_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RayIntegralOps<D>,
{
    let name = device.backend_name();

    // Uniform field: a +x ray through the middle has chord 16, so the
    // integral is 0.25 · 16 = 4; a ray far outside in y never samples the
    // volume and stays exactly zero.
    let uniform = build_field(|_, _, _| 0.25);
    let rays = [
        -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, // hit
        -10.0, 100.0, 4.0, 1.0, 0.0, 0.0, // miss
    ];
    let got = run(device, ops, &uniform, &rays, 0.5);
    assert!(
        (got[0] - 4.0).abs() < SUM_BOUND,
        "{name}: uniform chord integral {} != 4.0",
        got[0]
    );
    assert_eq!(got[1], 0.0, "{name}: a missing ray must integrate to 0");

    // Affine field f(x) = 0.005·x + 0.02 along the chord:
    // ∫₀¹⁶ f dx = 0.005·128 + 0.02·16 = 0.96, and midpoint sampling is exact
    // for affine integrands, so only summation error remains.
    let affine = build_field(|ix, _, _| 0.01 * ix as f32 + 0.02);
    let rays = [-10.0, 4.0, 4.0, 1.0, 0.0, 0.0];
    let got = run(device, ops, &affine, &rays, 1.0);
    assert!(
        (got[0] - 0.96).abs() < SUM_BOUND,
        "{name}: affine midpoint integral {} != 0.96",
        got[0]
    );

    // A uniform integral is independent of the step size.
    let coarse = run(device, ops, &uniform, &rays, 8.0)[0];
    let fine = run(device, ops, &uniform, &rays, 0.125)[0];
    assert!(
        (coarse - fine).abs() < SUM_BOUND,
        "{name}: step dependence: {coarse} vs {fine}"
    );

    // An output shorter than the ray count is rejected and untouched.
    let field = device.upload(&uniform).expect("field upload");
    let two_rays = device
        .upload(&[
            -10.0f32, 4.0, 4.0, 1.0, 0.0, 0.0, -10.0, 4.0, 4.0, 1.0, 0.0, 0.0,
        ])
        .expect("ray upload");
    let short_out = device.upload(&[9.0f32]).expect("output upload");
    let result =
        ops.ray_line_integrals_into(device, &field, geometry(), &two_rays, 0.5, &short_out);
    assert!(
        result.is_err(),
        "{name}: an output shorter than the ray count must be rejected"
    );
    let mut got = [0.0f32; 1];
    device.download(&short_out, &mut got).expect("download");
    assert_eq!(
        got,
        [9.0],
        "{name}: a rejected dispatch must not touch the output"
    );
}
