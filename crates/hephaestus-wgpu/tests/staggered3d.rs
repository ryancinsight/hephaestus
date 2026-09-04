#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Differential tests for the 3-D staggered gradient/divergence pair.
//!
//! The oracle is `leto_ops::StaggeredLeapfrog3D` — the same operator the CPU
//! backend runs — dispatched on a live device and compared value by value.
//! That comparison is the whole reason to trust the device divergence: it is a
//! gathered transpose derived by hand, including its wall closure, where the
//! CPU computes the transpose by scattering and gets the closure for free.
//!
//! The adjoint identity is checked independently, because agreeing with the CPU
//! and being an adjoint are different claims and a shared derivation mistake
//! could satisfy the first while failing the second.

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{Staggered3DKernel, Staggered3DParams, StaggeredAxis, WgpuDevice};
use leto::{Array3, ArrayView3, Layout};
use leto_ops::{Axis, StaggeredLeapfrog3D, staggered_first_derivative_coefficients};

fn device_or_skip() -> Option<WgpuDevice> {
    super::device_or_skip()
}

const SHAPE: [usize; 3] = [8, 6, 10];

fn cpu_axis(axis: StaggeredAxis) -> Axis {
    match axis {
        StaggeredAxis::X => Axis::X,
        StaggeredAxis::Y => Axis::Y,
        StaggeredAxis::Z => Axis::Z,
    }
}

fn row_major(shape: [usize; 3]) -> Layout<3> {
    Layout::<3>::try_new(
        shape,
        [(shape[1] * shape[2]) as isize, shape[2] as isize, 1],
        0,
    )
    .unwrap()
}

/// A non-separable field, so an axis or stride mistake cannot cancel out.
fn field(shape: [usize; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let x = i as f32 * 0.37;
                let y = j as f32 * 0.53;
                let z = k as f32 * 0.71;
                values.push(x.sin() * y.cos() + z.sin() * 0.75 + 0.25);
            }
        }
    }
    values
}

fn cpu_reference(
    values: &[f32],
    shape: [usize; 3],
    axis: StaggeredAxis,
    order: usize,
    spacing: [f32; 3],
    divergence: bool,
) -> Vec<f32> {
    let operator =
        StaggeredLeapfrog3D::<f32>::new(order, spacing[0], spacing[1], spacing[2]).unwrap();
    let view = ArrayView3::try_new(row_major(shape), values).unwrap();
    let mut out = Array3::<f32>::zeros(shape);
    if divergence {
        operator
            .divergence_into(cpu_axis(axis), view, &mut out.view_mut())
            .unwrap();
    } else {
        operator
            .gradient_into(cpu_axis(axis), view, &mut out.view_mut())
            .unwrap();
    }
    out.as_slice().unwrap().to_vec()
}

fn run_device(
    values: &[f32],
    shape: [usize; 3],
    axis: StaggeredAxis,
    order: usize,
    spacing: [f32; 3],
    divergence: bool,
) -> Option<Vec<f32>> {
    let device = device_or_skip()?;
    let taps = staggered_first_derivative_coefficients::<f32>(order / 2).unwrap();
    let params = Staggered3DParams::new(
        shape[0] as u32,
        shape[1] as u32,
        shape[2] as u32,
        axis,
        taps.taps(),
        spacing,
    )
    .unwrap();

    let input = device.upload(values).unwrap();
    let output = device.alloc_zeroed::<f32>(values.len()).unwrap();
    let kernel = Staggered3DKernel::new(&device).unwrap();
    if divergence {
        kernel
            .divergence(&device, &input, &output, &params)
            .unwrap();
    } else {
        kernel.gradient(&device, &input, &output, &params).unwrap();
    }

    let mut got = vec![0.0_f32; values.len()];
    device.download(&output, &mut got).unwrap();
    Some(got)
}

/// The stencil sums `2N` taps of a field of order one, so the difference from
/// the CPU is a few f32 roundings; the reduction order also differs between a
/// GPU thread and the CPU sweep, so exact equality is not the right claim.
fn assert_matches(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    let scale = expected.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
    let bound = 32.0 * f32::EPSILON * scale.max(1.0);
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= bound,
            "{what}: cell {index} device {got} vs cpu {expected} (bound {bound:e})"
        );
    }
}

fn check(axis: StaggeredAxis, order: usize, divergence: bool) {
    let values = field(SHAPE);
    let spacing = [1.5e-3_f32, 2.5e-3, 0.5e-3];
    let Some(got) = run_device(&values, SHAPE, axis, order, spacing, divergence) else {
        return;
    };
    let expected = cpu_reference(&values, SHAPE, axis, order, spacing, divergence);
    let what = if divergence { "divergence" } else { "gradient" };
    assert_matches(&got, &expected, &format!("{what} {axis:?} order {order}"));
}

pub(super) fn staggered_gradient_matches_cpu_on_every_axis() {
    for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
        check(axis, 2, false);
        check(axis, 4, false);
    }
}

pub(super) fn staggered_divergence_matches_cpu_on_every_axis() {
    for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
        check(axis, 2, true);
        check(axis, 4, true);
    }
}

pub(super) fn staggered_high_order_matches_cpu() {
    for divergence in [false, true] {
        check(StaggeredAxis::Z, 6, divergence);
        check(StaggeredAxis::X, 8, divergence);
    }
}

/// The wall closure is the part the gathered transpose had to re-derive, so it
/// is checked where it bites: a field that is constant along the swept axis has
/// a gradient of exactly zero everywhere, walls included, under reflection.
pub(super) fn a_field_constant_along_the_axis_has_no_device_gradient() {
    for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
        let values = vec![2.75_f32; SHAPE[0] * SHAPE[1] * SHAPE[2]];
        let Some(got) = run_device(&values, SHAPE, axis, 4, [1.0, 1.0, 1.0], false) else {
            return;
        };
        for (index, value) in got.iter().enumerate() {
            assert_eq!(*value, 0.0, "{axis:?}: cell {index} is {value}");
        }
    }
}

/// `D = -Gᵀ` on the device itself, not inherited from the CPU comparison: a
/// derivation mistake shared by both operators could match the CPU and still
/// break the identity a conservative leapfrog rests on.
pub(super) fn the_device_pair_is_a_negative_adjoint() {
    for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
        let p = field(SHAPE);
        let u: Vec<f32> = p.iter().rev().map(|v| v * 0.6 + 0.1).collect();
        let spacing = [1.0_f32, 1.0, 1.0];

        let Some(grad_p) = run_device(&p, SHAPE, axis, 4, spacing, false) else {
            return;
        };
        let div_u = run_device(&u, SHAPE, axis, 4, spacing, true).unwrap();

        let left: f32 = grad_p.iter().zip(&u).map(|(a, b)| a * b).sum();
        let right: f32 = -p.iter().zip(&div_u).map(|(a, b)| a * b).sum::<f32>();
        let cells = p.len() as f32;
        let bound = 64.0 * f32::EPSILON * left.abs().max(right.abs()).max(1.0) * cells;
        assert!(
            (left - right).abs() <= bound,
            "{axis:?}: <Gp,u> {left:e} vs -<p,Du> {right:e} (bound {bound:e})"
        );
        assert!(
            left.abs() > 1e-3,
            "{axis:?}: the identity held trivially, inner product {left:e}"
        );
    }
}

/// A grid thinner than the stencil is rejected when the parameters are built,
/// not swept with a reflection the kernel cannot resolve in one step.
pub(super) fn a_grid_thinner_than_the_stencil_is_rejected() {
    let taps = staggered_first_derivative_coefficients::<f32>(3).unwrap();
    // Order six needs six cells on the swept axis; give it five.
    assert!(
        Staggered3DParams::new(5, 8, 8, StaggeredAxis::X, taps.taps(), [1.0, 1.0, 1.0]).is_err()
    );
    // The same grid is fine on an axis that is deep enough.
    assert!(
        Staggered3DParams::new(5, 8, 8, StaggeredAxis::Y, taps.taps(), [1.0, 1.0, 1.0]).is_ok()
    );
}

pub(super) fn storage_length_mismatch_is_rejected_before_launch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let taps = staggered_first_derivative_coefficients::<f32>(1).unwrap();
    let params =
        Staggered3DParams::new(4, 4, 4, StaggeredAxis::Z, taps.taps(), [1.0, 1.0, 1.0]).unwrap();
    let input = device.upload(&vec![0.0_f32; 63]).unwrap();
    let output = device.alloc_zeroed::<f32>(64).unwrap();
    let kernel = Staggered3DKernel::new(&device).unwrap();
    assert!(kernel.gradient(&device, &input, &output, &params).is_err());
}
