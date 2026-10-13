#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Differential tests for the 2D Laplacian stencil kernel.
//!
//! The provider-owned kernel is dispatched on a live wgpu device and its
//! output is compared against Leto's CPU implementation of the same typed
//! stencil, including Dirichlet/Neumann/Periodic boundary handling.

use aequitas::systems::si::{quantities::Length, units::Meter};
use eunomia::FloatElement;
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{
    BoundaryCondition, Laplacian2DKernel, Laplacian2DParams, LaplacianPolarity, WgpuDevice,
};
use leto::{Array1, Laplacian2D};
use leto_ops::laplacian_2d_into;

fn device_or_skip() -> Option<WgpuDevice> {
    super::device_or_skip()
}

/// Leto CPU reference for the same provider-owned stencil used by the WGSL
/// kernel.
fn leto_laplacian_2d(
    field: &[f32],
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
    bc: BoundaryCondition,
) -> Vec<f32> {
    let input = Array1::from_shape_vec([field.len()], field.to_vec()).expect("valid input shape");
    let mut output = Array1::zeros([field.len()]);
    let stencil = Laplacian2D::new(
        nx,
        ny,
        Length::from_unit::<Meter>(dx),
        Length::from_unit::<Meter>(dy),
        bc,
    )
    .expect("valid stencil contract");
    laplacian_2d_into(&stencil, &input.view(), &mut output.view_mut())
        .expect("matching stencil storage");
    output.iter().copied().collect()
}

fn run_laplacian(
    field: &[f32],
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
    bc: BoundaryCondition,
) -> Option<Vec<f32>> {
    let device = device_or_skip()?;

    let input = device.upload(field).unwrap();
    let output = device.alloc_zeroed::<f32>(field.len()).unwrap();
    let params = Laplacian2DParams::new(
        u32::try_from(nx).expect("fixture x extent fits device"),
        u32::try_from(ny).expect("fixture y extent fits device"),
        Length::from_unit::<Meter>(dx),
        Length::from_unit::<Meter>(dy),
        bc,
        LaplacianPolarity::Laplacian,
    )
    .unwrap();
    let kernel = Laplacian2DKernel::new(&device).unwrap();
    kernel.dispatch(&device, &input, &output, &params).unwrap();

    let mut got = vec![0.0f32; field.len()];
    device.download(&output, &mut got).unwrap();
    Some(got)
}

pub(super) fn laplacian_minimum_grid_matches_cpu_reference() {
    let field: Vec<f32> = (0..4).map(|i| f32::from_count(i) * 0.5 - 0.75).collect();
    let Some(got) = run_laplacian(&field, 2, 2, 1.0, 1.0, BoundaryCondition::Dirichlet) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, 2, 2, 1.0, 1.0, BoundaryCondition::Dirichlet);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_dirichlet_matches_cpu_reference() {
    let Some(got) = run_laplacian(
        &(0..30).map(f32::from_count).collect::<Vec<_>>(),
        6,
        5,
        0.5,
        1.0,
        BoundaryCondition::Dirichlet,
    ) else {
        return;
    };
    let expected = leto_laplacian_2d(
        &(0..30).map(f32::from_count).collect::<Vec<_>>(),
        6,
        5,
        0.5,
        1.0,
        BoundaryCondition::Dirichlet,
    );
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_neumann_matches_cpu_reference() {
    let field: Vec<f32> = (0..30)
        .map(|k| {
            let i = f32::from_count(k % 6);
            let j = f32::from_count(k / 6);
            (i + 1.0).ln() + (j + 2.0).sin()
        })
        .collect();

    let Some(got) = run_laplacian(&field, 6, 5, 0.25, 0.75, BoundaryCondition::Neumann) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, 6, 5, 0.25, 0.75, BoundaryCondition::Neumann);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_periodic_matches_cpu_reference() {
    let field: Vec<f32> = (0..30)
        .map(|k| {
            let i = f32::from_count(k % 6);
            let j = f32::from_count(k / 6);
            i.sin() + j.cos()
        })
        .collect();

    let Some(got) = run_laplacian(&field, 6, 5, 1.0, 1.0, BoundaryCondition::Periodic) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, 6, 5, 1.0, 1.0, BoundaryCondition::Periodic);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_non_square_2x3_matches_cpu_reference() {
    // Covers nx < ny aspect ratio and the nx == 2 minimum in the X direction.
    let field: Vec<f32> = (0..6)
        .map(|k| {
            let i = f32::from_count(k % 2);
            let j = f32::from_count(k / 2);
            i * i + j * j * j
        })
        .collect();

    let Some(got) = run_laplacian(&field, 2, 3, 0.5, 0.75, BoundaryCondition::Dirichlet) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, 2, 3, 0.5, 0.75, BoundaryCondition::Dirichlet);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_non_square_3x2_matches_cpu_reference() {
    // Covers ny < nx aspect ratio and the ny == 2 minimum in the Y direction.
    let field: Vec<f32> = (0..6)
        .map(|k| {
            let i = f32::from_count(k % 3);
            let j = f32::from_count(k / 3);
            (i + 1.0).ln() + (j + 2.0).sin()
        })
        .collect();

    let Some(got) = run_laplacian(&field, 3, 2, 0.25, 1.0, BoundaryCondition::Neumann) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, 3, 2, 0.25, 1.0, BoundaryCondition::Neumann);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_large_dirichlet_16x16_matches_cpu_reference() {
    // Exercises multiple 8x8 workgroups with Dirichlet boundaries.
    let n = 16;
    let field: Vec<f32> = (0..n * n)
        .map(|k| {
            let i = f32::from_count(k % n);
            let j = f32::from_count(k / n);
            (i + 1.0).ln() + (j + 2.0).sin()
        })
        .collect();

    let Some(got) = run_laplacian(&field, n, n, 0.125, 0.25, BoundaryCondition::Dirichlet) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, n, n, 0.125, 0.25, BoundaryCondition::Dirichlet);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_large_periodic_16x16_matches_cpu_reference() {
    // Exercises multiple 8x8 workgroups with a genuinely periodic field.
    let n = 16;
    let field: Vec<f32> = (0..n * n)
        .map(|k| {
            let i = f32::from_count(k % n);
            let j = f32::from_count(k / n);
            let x = 2.0 * std::f32::consts::PI * i / f32::from_count(n);
            let y = 2.0 * std::f32::consts::PI * j / f32::from_count(n);
            x.sin() * y.cos()
        })
        .collect();

    let Some(got) = run_laplacian(&field, n, n, 0.125, 0.25, BoundaryCondition::Periodic) else {
        return;
    };
    let expected = leto_laplacian_2d(&field, n, n, 0.125, 0.25, BoundaryCondition::Periodic);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

pub(super) fn laplacian_storage_length_mismatch_is_rejected_before_launch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let input = device.upload(&[0.0f32; 29]).expect("input upload");
    let output = device.alloc_zeroed::<f32>(30).expect("output allocation");
    let params = Laplacian2DParams::new(
        6,
        5,
        Length::from_unit::<Meter>(1.0),
        Length::from_unit::<Meter>(1.0),
        BoundaryCondition::Dirichlet,
        LaplacianPolarity::Laplacian,
    )
    .expect("valid parameters");
    let kernel = Laplacian2DKernel::new(&device).expect("kernel compile");
    let error = kernel
        .dispatch(&device, &input, &output, &params)
        .expect_err("storage mismatch must fail");
    assert!(matches!(error, HephaestusError::LengthMismatch { .. }));
}

fn assert_close_slice(got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        let tolerance = abs_tol.max(rel_tol * expected.abs().max(1.0));
        assert!(
            (got - expected).abs() <= tolerance,
            "slice mismatch at {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
}
