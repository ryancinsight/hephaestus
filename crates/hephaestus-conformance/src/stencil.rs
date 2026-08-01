//! Contract clauses for the device-neutral 2D stencil seam.
//!
//! The oracle is analytical: the 5-point Laplacian of the quadratic field
//! `f(x, y) = x² + y²` on a unit grid is exactly `4` at every interior
//! point — finite differences of quadratics are exact, and every
//! intermediate value is a small integer, so `f32` arithmetic is exact
//! throughout.

use aequitas::systems::si::{quantities::Length, units::Meter};
use hephaestus_core::{
    BoundaryCondition, ComputeDevice, Laplacian2DParams, LaplacianPolarity, StencilOps,
};

/// Run every stencil clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_stencil_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StencilOps<D>,
{
    let name = device.backend_name();
    let kernel = ops.prepare_laplacian_2d(device).expect("kernel compile");

    // f(x, y) = x² + y² on a 4x4 unit grid, row-major with x fastest.
    let mut field = [0.0f32; 16];
    for (index, value) in field.iter_mut().enumerate() {
        let x = (index % 4) as f32;
        let y = (index / 4) as f32;
        *value = x * x + y * y;
    }
    let input = device.upload(&field).expect("field upload");
    let output = device.alloc_zeroed::<f32>(16).expect("output alloc");
    let params = Laplacian2DParams::new(
        4,
        4,
        Length::from_unit::<Meter>(1.0),
        Length::from_unit::<Meter>(1.0),
        BoundaryCondition::Dirichlet,
        LaplacianPolarity::Laplacian,
    )
    .expect("params");

    ops.laplacian_2d_into(device, &kernel, &input, &output, &params)
        .expect("laplacian dispatch");
    let mut got = [0.0f32; 16];
    device.download(&output, &mut got).expect("download");
    for y in 1..3 {
        for x in 1..3 {
            let value = got[y * 4 + x];
            assert_eq!(
                value, 4.0,
                "{name}: interior Laplacian of x²+y² at ({x},{y}) must be exactly 4"
            );
        }
    }

    // The compiled kernel is reusable: a second dispatch over the same
    // operands reproduces the result exactly.
    ops.laplacian_2d_into(device, &kernel, &input, &output, &params)
        .expect("laplacian re-dispatch");
    let mut again = [0.0f32; 16];
    device.download(&output, &mut again).expect("download");
    assert_eq!(got, again, "{name}: kernel reuse must be deterministic");

    // A storage-length mismatch is rejected.
    let short = device.alloc_zeroed::<f32>(8).expect("short alloc");
    assert!(
        ops.laplacian_2d_into(device, &kernel, &input, &short, &params)
            .is_err(),
        "{name}: a short output buffer must be rejected"
    );
}
