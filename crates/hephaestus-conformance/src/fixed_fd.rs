//! Contract clauses for the device-neutral fixed-scheme 3-D sweep seam.
//!
//! Three oracles, each catching what the others cannot:
//!
//! - **Provider.** The device dispatch runs lane-by-lane against
//!   `leto_ops::FiniteDifference3D` on the same field — the CPU
//!   implementation this seam exists to mirror — over every scheme, every
//!   axis, a general shape, and each scheme's minimum axis extent, so every
//!   boundary fall-back branch is exercised. The kernels reproduce the
//!   provider's operation order with its exact scales, so the bound is a few
//!   ULPs of shader-compiler reassociation, not a modeling tolerance.
//! - **Analytical.** A ramp of slope `s` along the swept axis differentiates
//!   to `s` at every lane under every scheme, walls included: each closure
//!   is exact on a linear field. That falls out of the stencil weights
//!   rather than of any implementation.
//! - **Structural.** A constant field differentiates to exactly zero
//!   everywhere, which is what the one-sided wall closures mean; a wall
//!   handled as zero-extension instead would show a step.

use hephaestus_core::{
    ComputeDevice, FixedFd3DOps, FixedFd3DParams, FixedFd3DScheme, StaggeredAxis,
};
use leto::{ArrayView3, ArrayViewMut3, Layout as LetoLayout};
use leto_ops::{Axis, FiniteDifference3D, FiniteDifference3DScheme as LetoScheme};

const AXES: [StaggeredAxis; 3] = [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z];
const SCHEMES: [FixedFd3DScheme; 5] = [
    FixedFd3DScheme::CentralSecondOrder,
    FixedFd3DScheme::CentralFourthOrder,
    FixedFd3DScheme::CentralSixthOrder,
    FixedFd3DScheme::StaggeredForward,
    FixedFd3DScheme::StaggeredBackward,
];
/// General shape: every extent clears every scheme's minimum, so all five
/// schemes sweep all three axes on it.
const SHAPE: [u32; 3] = [8, 7, 10];
const SPACING: f32 = 0.5;

fn map_scheme(scheme: FixedFd3DScheme) -> LetoScheme {
    match scheme {
        FixedFd3DScheme::CentralSecondOrder => LetoScheme::CentralSecondOrder,
        FixedFd3DScheme::CentralFourthOrder => LetoScheme::CentralFourthOrder,
        FixedFd3DScheme::CentralSixthOrder => LetoScheme::CentralSixthOrder,
        FixedFd3DScheme::StaggeredForward => LetoScheme::StaggeredForward,
        FixedFd3DScheme::StaggeredBackward => LetoScheme::StaggeredBackward,
    }
}

fn map_axis(axis: StaggeredAxis) -> Axis {
    match axis {
        StaggeredAxis::X => Axis::X,
        StaggeredAxis::Y => Axis::Y,
        StaggeredAxis::Z => Axis::Z,
    }
}

/// Axis extents each scheme must additionally prove: its minimum, where every
/// fall-back branch is live. Fourth order also proves the singleton (flat)
/// and two-point (one-sided at both ends) axes it alone accepts.
fn extra_extents(scheme: FixedFd3DScheme) -> &'static [u32] {
    match scheme {
        FixedFd3DScheme::CentralSecondOrder => &[3],
        FixedFd3DScheme::CentralFourthOrder => &[1, 2],
        FixedFd3DScheme::CentralSixthOrder => &[7],
        FixedFd3DScheme::StaggeredForward | FixedFd3DScheme::StaggeredBackward => &[2],
    }
}

fn coordinate(index: usize, shape: [usize; 3], axis: StaggeredAxis) -> usize {
    let [_, ny, nz] = shape;
    match axis {
        StaggeredAxis::X => index / (ny * nz),
        StaggeredAxis::Y => (index / nz) % ny,
        StaggeredAxis::Z => index % nz,
    }
}

fn dense_layout(shape: [usize; 3]) -> LetoLayout<3> {
    let [nx, ny, nz] = shape;
    let strides = [
        isize::try_from(ny * nz).expect("a stride fitting isize"),
        isize::try_from(nz).expect("a stride fitting isize"),
        1,
    ];
    LetoLayout::<3>::try_new([nx, ny, nz], strides, 0).expect("a contiguous layout")
}

fn oracle(
    scheme: FixedFd3DScheme,
    axis: StaggeredAxis,
    shape: [usize; 3],
    field: &[f32],
) -> Vec<f32> {
    let operator = FiniteDifference3D::new(map_scheme(scheme), SPACING, SPACING, SPACING)
        .expect("the provider accepts a positive spacing");
    let [nx, ny, nz] = shape;
    let input_layout = dense_layout([nx, ny, nz]);
    let mut out_shape = shape;
    if !scheme.preserves_shape() {
        out_shape[axis.index()] -= 1;
    }
    let output_layout = dense_layout(out_shape);
    let view = ArrayView3::try_new(input_layout, field).expect("an input view over the field");
    let mut expected = vec![0.0_f32; out_shape.iter().product()];
    let mut target =
        ArrayViewMut3::try_new(output_layout, &mut expected).expect("an output view over the lane");
    match map_axis(axis) {
        Axis::X => operator.apply_x_into(view, &mut target),
        Axis::Y => operator.apply_y_into(view, &mut target),
        Axis::Z => operator.apply_z_into(view, &mut target),
    }
    .expect("the provider sweeps a valid grid");
    expected
}

fn dispatch<D, S>(
    device: &D,
    ops: &S,
    kernel: &S::FixedFd3D,
    params: &FixedFd3DParams,
    field: &[f32],
) -> Vec<f32>
where
    D: ComputeDevice,
    S: FixedFd3DOps<D>,
{
    let out_cells = params.output_cell_count().expect("a countable output grid");
    let input = device.upload(field).expect("field upload");
    let output = device.alloc_zeroed::<f32>(out_cells).expect("output alloc");
    ops.fixed_fd_into(device, kernel, &input, &output, params)
        .expect("sweep dispatch");
    let mut got = vec![0.0_f32; out_cells];
    device.download(&output, &mut got).expect("sweep readback");
    got
}

/// Run every fixed-scheme clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a device.
pub fn assert_fixed_fd_3d_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: FixedFd3DOps<D>,
{
    let name = device.backend_name();
    let kernel = ops
        .prepare_fixed_fd_3d(device)
        .expect("sweep kernel compile");

    for scheme in SCHEMES {
        for axis in AXES {
            let mut shapes = vec![SHAPE.map(|extent| extent as usize)];
            for extent in extra_extents(scheme) {
                let mut shape = [5_usize, 4, 6];
                shape[axis.index()] = *extent as usize;
                shapes.push(shape);
            }
            for shape in shapes {
                let [nx, ny, nz] = shape;
                let params = FixedFd3DParams::new(
                    nx as u32,
                    ny as u32,
                    nz as u32,
                    axis,
                    scheme,
                    [SPACING; 3],
                )
                .expect("a valid parameter block");
                let count = nx * ny * nz;
                let tag = format!("{name}: {scheme:?} on {axis:?} over {shape:?}");

                // Provider: the device sweep matches the CPU sweep lane by lane.
                let field: Vec<f32> = (0..count)
                    .map(|index| {
                        ((index * 7 % 13) as f32).mul_add(0.31, -((index * 3 % 7) as f32) * 0.17)
                    })
                    .collect();
                let expected = oracle(scheme, axis, shape, &field);
                let got = dispatch(device, ops, &kernel, &params, &field);
                assert_eq!(got.len(), expected.len(), "{tag}: lane count");
                for (index, (value, reference)) in got.iter().zip(&expected).enumerate() {
                    let deviation = (value - reference).abs();
                    let bound = 32.0 * f32::EPSILON * reference.abs().max(1.0);
                    assert!(
                        deviation <= bound,
                        "{tag}: lane {index} differentiates to {value:e}, provider says \
                         {reference:e} (bound {bound:e})"
                    );
                }

                // Analytical: a ramp differentiates to its slope everywhere.
                // A singleton axis has no variation along it, so fourth order
                // is flat there by design and the provider clause above is the
                // only oracle for that shape.
                if shape[axis.index()] > 1 {
                    let slope = 3.0_f32;
                    let ramp: Vec<f32> = (0..count)
                        .map(|index| slope * SPACING * coordinate(index, shape, axis) as f32)
                        .collect();
                    let got = dispatch(device, ops, &kernel, &params, &ramp);
                    for (index, value) in got.iter().enumerate() {
                        let bound = 8.0 * f32::EPSILON * slope;
                        assert!(
                            (value - slope).abs() <= bound,
                            "{tag}: lane {index} of a ramp of slope {slope} differentiates to \
                             {value:e} (bound {bound:e})"
                        );
                    }
                }

                // Structural: a constant field is flat everywhere, walls included.
                let flat = vec![-1.25_f32; count];
                let got = dispatch(device, ops, &kernel, &params, &flat);
                for (index, value) in got.iter().enumerate() {
                    assert!(
                        value.abs() <= f32::EPSILON,
                        "{tag}: a constant field has derivative {value:e} at lane {index}"
                    );
                }
            }
        }
    }
}
