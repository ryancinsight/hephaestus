//! Contract clauses for the device-neutral 3-D staggered seam.
//!
//! Three oracles, each catching what the others cannot:
//!
//! - **Analytical.** A field linear along the swept axis has a staggered
//!   gradient exactly equal to its slope at every interior face, at every
//!   order. That falls straight out of the coefficient condition
//!   `Σ c_n (n − ½) = ½`, so it is a property of the stencil rather than of any
//!   implementation, and it is exact in `f32` for the small integers used here.
//! - **Structural.** A field constant along the axis has a gradient of exactly
//!   zero everywhere including the walls, which is what the reflected closure
//!   means; a wall handled as zero-extension instead would show a step.
//! - **Algebraic.** `D = -Gᵀ` on the device's own pair. A backend whose
//!   divergence gathers rather than scatters re-derives the transpose, and this
//!   is the clause that says whether it re-derived it correctly.

use hephaestus_core::{ComputeDevice, Staggered3DOps, Staggered3DParams, StaggeredAxis};
use leto_ops::staggered_first_derivative_coefficients;

const AXES: [StaggeredAxis; 3] = [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z];
const SHAPE: [u32; 3] = [8, 6, 10];

fn cells() -> usize {
    SHAPE.iter().map(|extent| *extent as usize).product()
}

fn coordinate(index: usize, axis: StaggeredAxis) -> usize {
    let (ny, nz) = (SHAPE[1] as usize, SHAPE[2] as usize);
    match axis {
        StaggeredAxis::X => index / (ny * nz),
        StaggeredAxis::Y => (index / nz) % ny,
        StaggeredAxis::Z => index % nz,
    }
}

fn params(axis: StaggeredAxis, order: usize, spacing: f32) -> Staggered3DParams {
    let taps = staggered_first_derivative_coefficients::<f32>(order / 2)
        .expect("the provider derives taps for a supported order");
    Staggered3DParams::new(
        SHAPE[0],
        SHAPE[1],
        SHAPE[2],
        axis,
        taps.taps(),
        [spacing; 3],
    )
    .expect("a grid deeper than the stencil")
}

/// Run every staggered clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a device.
pub fn assert_staggered_3d_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: Staggered3DOps<D>,
{
    let name = device.backend_name();
    let kernel = ops
        .prepare_staggered_3d(device)
        .expect("staggered kernel compile");
    let count = cells();

    for axis in AXES {
        for order in [2_usize, 4] {
            let spacing = 0.5_f32;
            let params = params(axis, order, spacing);
            let halo = params.half_order() as usize;
            let extent = SHAPE[axis.index()] as usize;

            // Analytical: a ramp of slope `slope` along the axis.
            let slope = 3.0_f32;
            let field: Vec<f32> = (0..count)
                .map(|index| slope * spacing * coordinate(index, axis) as f32)
                .collect();
            let input = device.upload(&field).expect("ramp upload");
            let output = device.alloc_zeroed::<f32>(count).expect("output alloc");
            ops.staggered_gradient_into(device, &kernel, &input, &output, &params)
                .expect("gradient dispatch");
            let mut got = vec![0.0_f32; count];
            device
                .download(&output, &mut got)
                .expect("gradient readback");

            for (index, value) in got.iter().enumerate() {
                let coordinate = coordinate(index, axis);
                // Faces within the halo of either wall read reflected taps, so
                // the ramp is not linear across their stencil.
                if coordinate + 1 < halo || coordinate + halo >= extent {
                    continue;
                }
                assert!(
                    (value - slope).abs() <= 1e-4 * slope,
                    "{name}: {axis:?} order {order}: interior face {index} of a ramp of slope \
                     {slope} differentiates to {value}"
                );
            }

            // Structural: a constant field is flat everywhere, walls included.
            let flat = vec![-1.25_f32; count];
            let input = device.upload(&flat).expect("constant upload");
            let output = device.alloc_zeroed::<f32>(count).expect("output alloc");
            ops.staggered_gradient_into(device, &kernel, &input, &output, &params)
                .expect("gradient dispatch");
            let mut got = vec![0.0_f32; count];
            device
                .download(&output, &mut got)
                .expect("gradient readback");
            for (index, value) in got.iter().enumerate() {
                assert!(
                    value.abs() <= f32::EPSILON,
                    "{name}: {axis:?} order {order}: a constant field has gradient {value} at \
                     cell {index}; a reflected wall would give exactly zero"
                );
            }
        }

        // Algebraic: the pair is a negative adjoint.
        let order = 4_usize;
        let params = params(axis, order, 1.0);
        let p: Vec<f32> = (0..count)
            .map(|index| ((index % 7) as f32).mul_add(0.3, ((index % 5) as f32) * -0.11))
            .collect();
        let u: Vec<f32> = (0..count)
            .map(|index| ((index % 11) as f32).mul_add(0.17, 0.4))
            .collect();

        let mut gradient = vec![0.0_f32; count];
        let input = device.upload(&p).expect("p upload");
        let output = device.alloc_zeroed::<f32>(count).expect("output alloc");
        ops.staggered_gradient_into(device, &kernel, &input, &output, &params)
            .expect("gradient dispatch");
        device
            .download(&output, &mut gradient)
            .expect("gradient readback");

        let mut divergence = vec![0.0_f32; count];
        let input = device.upload(&u).expect("u upload");
        let output = device.alloc_zeroed::<f32>(count).expect("output alloc");
        ops.staggered_divergence_into(device, &kernel, &input, &output, &params)
            .expect("divergence dispatch");
        device
            .download(&output, &mut divergence)
            .expect("divergence readback");

        let left: f32 = gradient.iter().zip(&u).map(|(a, b)| a * b).sum();
        let right: f32 = -p.iter().zip(&divergence).map(|(a, b)| a * b).sum::<f32>();
        // Both sides sum the same products in a different order, so the bound is
        // the accumulated rounding of a length-N f32 sum.
        let bound = 64.0 * f32::EPSILON * left.abs().max(right.abs()).max(1.0) * count as f32;
        assert!(
            (left - right).abs() <= bound,
            "{name}: {axis:?}: the pair is not a negative adjoint, <Gp,u>={left:e} but \
             -<p,Du>={right:e} (bound {bound:e})"
        );
        assert!(
            left.abs() > 1e-3,
            "{name}: {axis:?}: the adjoint clause held trivially, inner product {left:e}"
        );
    }
}
