//! CUDA differential contracts for the 3-D staggered gradient/divergence pair.
//!
//! The oracle is `leto_ops::StaggeredLeapfrog3D` — the operator the CPU backend
//! runs — launched on a live device and compared value by value. That is the
//! reason to trust the CUDA divergence: like the WGSL one it gathers a
//! transpose derived by hand, including the wall closure the CPU gets for free
//! by scattering.
//!
//! The shared conformance clauses run beside the differential, so the CUDA
//! backend answers the same three oracles the WGPU backend does.

use hephaestus_conformance::assert_staggered_3d_contract;
use hephaestus_core::ComputeDevice;
use hephaestus_cuda::{
    CudaDevice, CudaStaggered3DOps, Staggered3DKernel, Staggered3DParams, StaggeredAxis,
};
use leto::{Array3, ArrayView3, Layout};
use leto_ops::{Axis, StaggeredLeapfrog3D, staggered_first_derivative_coefficients};

const SHAPE: [usize; 3] = [8, 8, 10];
const AXES: [StaggeredAxis; 3] = [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z];

fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_some() {
                panic!("CUDA device required for {test}: {error}");
            }
            eprintln!("skipping CUDA staggered contract {test}: {error}");
            None
        }
    }
}

fn cpu_axis(axis: StaggeredAxis) -> Axis {
    match axis {
        StaggeredAxis::X => Axis::X,
        StaggeredAxis::Y => Axis::Y,
        StaggeredAxis::Z => Axis::Z,
    }
}

fn row_major() -> Layout<3> {
    Layout::<3>::try_new(
        SHAPE,
        [(SHAPE[1] * SHAPE[2]) as isize, SHAPE[2] as isize, 1],
        0,
    )
    .expect("row-major layout over a non-empty shape")
}

/// A non-separable field, so an axis or stride mistake cannot cancel out.
fn field() -> Vec<f32> {
    let mut values = Vec::with_capacity(SHAPE[0] * SHAPE[1] * SHAPE[2]);
    for i in 0..SHAPE[0] {
        for j in 0..SHAPE[1] {
            for k in 0..SHAPE[2] {
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
    axis: StaggeredAxis,
    order: usize,
    spacing: [f32; 3],
    divergence: bool,
) -> Vec<f32> {
    let operator = StaggeredLeapfrog3D::<f32>::new(order, spacing[0], spacing[1], spacing[2])
        .expect("supported order and positive spacing");
    let view = ArrayView3::try_new(row_major(), values).expect("view over the grid");
    let mut out = Array3::<f32>::zeros(SHAPE);
    if divergence {
        operator
            .divergence_into(cpu_axis(axis), view, &mut out.view_mut())
            .expect("matching destination shape");
    } else {
        operator
            .gradient_into(cpu_axis(axis), view, &mut out.view_mut())
            .expect("matching destination shape");
    }
    out.as_slice().expect("contiguous output").to_vec()
}

fn params(axis: StaggeredAxis, order: usize, spacing: [f32; 3]) -> Staggered3DParams {
    let taps = staggered_first_derivative_coefficients::<f32>(order / 2)
        .expect("the provider derives taps for a supported order");
    Staggered3DParams::new(
        SHAPE[0] as u32,
        SHAPE[1] as u32,
        SHAPE[2] as u32,
        axis,
        taps.taps(),
        spacing,
    )
    .expect("a grid deeper than the stencil")
}

fn run_device(
    device: &CudaDevice,
    kernel: &Staggered3DKernel,
    values: &[f32],
    axis: StaggeredAxis,
    order: usize,
    spacing: [f32; 3],
    divergence: bool,
) -> Vec<f32> {
    let params = params(axis, order, spacing);
    let input = device.upload(values).expect("input upload");
    let output = device
        .alloc_zeroed::<f32>(values.len())
        .expect("output allocation");
    if divergence {
        kernel
            .divergence(device, &input, &output, &params)
            .expect("divergence launch");
    } else {
        kernel
            .gradient(device, &input, &output, &params)
            .expect("gradient launch");
    }
    let mut got = vec![0.0_f32; values.len()];
    device.download(&output, &mut got).expect("readback");
    got
}

/// The stencil sums `2N` taps, and a CUDA thread's accumulation order differs
/// from the CPU sweep's, so the claim is an epsilon bound rather than bitwise
/// equality (reduction-order sensitivity).
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

#[test]
fn gradient_and_divergence_match_cpu_on_every_axis_and_order() {
    let Some(device) = device("gradient_and_divergence_match_cpu_on_every_axis_and_order") else {
        return;
    };
    let kernel = Staggered3DKernel::new(&device).expect("kernel compile");
    let values = field();
    let spacing = [1.5e-3_f32, 2.5e-3, 0.5e-3];

    for axis in AXES {
        for order in [2_usize, 4, 6, 8] {
            for divergence in [false, true] {
                let got = run_device(&device, &kernel, &values, axis, order, spacing, divergence);
                let expected = cpu_reference(&values, axis, order, spacing, divergence);
                let operator = if divergence { "divergence" } else { "gradient" };
                assert_matches(
                    &got,
                    &expected,
                    &format!("{operator} {axis:?} order {order}"),
                );
            }
        }
    }
}

/// The wall closure is what the gathered transpose had to re-derive, so it is
/// checked where it bites: a field constant along the swept axis differentiates
/// to exactly zero everywhere under reflection, walls included.
#[test]
fn a_field_constant_along_the_axis_has_no_gradient() {
    let Some(device) = device("a_field_constant_along_the_axis_has_no_gradient") else {
        return;
    };
    let kernel = Staggered3DKernel::new(&device).expect("kernel compile");
    let values = vec![2.75_f32; SHAPE[0] * SHAPE[1] * SHAPE[2]];
    for axis in AXES {
        let got = run_device(&device, &kernel, &values, axis, 4, [1.0, 1.0, 1.0], false);
        for (index, value) in got.iter().enumerate() {
            assert_eq!(*value, 0.0, "{axis:?}: cell {index} is {value}");
        }
    }
}

/// `D = -Gᵀ` measured on the device's own outputs: a derivation mistake shared
/// by both operators could match the CPU and still break the identity a
/// conservative leapfrog rests on.
#[test]
fn the_device_pair_is_a_negative_adjoint() {
    let Some(device) = device("the_device_pair_is_a_negative_adjoint") else {
        return;
    };
    let kernel = Staggered3DKernel::new(&device).expect("kernel compile");
    let p = field();
    let u: Vec<f32> = p.iter().rev().map(|v| v * 0.6 + 0.1).collect();
    let spacing = [1.0_f32, 1.0, 1.0];

    for axis in AXES {
        let gradient = run_device(&device, &kernel, &p, axis, 4, spacing, false);
        let divergence = run_device(&device, &kernel, &u, axis, 4, spacing, true);

        let left: f32 = gradient.iter().zip(&u).map(|(a, b)| a * b).sum();
        let right: f32 = -p.iter().zip(&divergence).map(|(a, b)| a * b).sum::<f32>();
        // Both sides sum the same products in a different order, so the bound
        // is the accumulated rounding of a length-N f32 sum.
        let bound = 64.0 * f32::EPSILON * left.abs().max(right.abs()).max(1.0) * p.len() as f32;
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

#[test]
fn cuda_satisfies_the_staggered_contract() {
    let Some(device) = device("cuda_satisfies_the_staggered_contract") else {
        return;
    };
    assert_staggered_3d_contract(&device, &CudaStaggered3DOps);
}

#[test]
fn storage_length_mismatch_is_rejected_before_launch() {
    let Some(device) = device("storage_length_mismatch_is_rejected_before_launch") else {
        return;
    };
    let kernel = Staggered3DKernel::new(&device).expect("kernel compile");
    let params = params(StaggeredAxis::Z, 2, [1.0, 1.0, 1.0]);
    let short = vec![0.0_f32; SHAPE[0] * SHAPE[1] * SHAPE[2] - 1];
    let input = device.upload(&short).expect("input upload");
    let output = device
        .alloc_zeroed::<f32>(SHAPE[0] * SHAPE[1] * SHAPE[2])
        .expect("output allocation");
    assert!(kernel.gradient(&device, &input, &output, &params).is_err());
}
