//! Differential tests for the 2D Laplacian stencil kernel.
//!
//! The provider-owned kernel is dispatched on a live wgpu device and its
//! output is compared against a CPU replica of the same finite-difference
//! stencil, including Dirichlet/Neumann/Periodic boundary handling.

use aequitas::systems::si::{quantities::Length, units::Meter};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{BoundaryCondition, Laplacian2DKernel, Laplacian2DParams, WgpuDevice};

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-stencil-laplacian-test") {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping Laplacian stencil test: {error}");
            None
        }
    }
}

/// CPU reference for the same 5-point Laplacian stencil used by the WGSL
/// kernel.  The implementation mirrors the kernel's boundary handling so the
/// test is a true differential oracle.
fn laplacian_2d_reference(
    field: &[f32],
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
    bc: BoundaryCondition,
) -> Vec<f32> {
    assert_eq!(field.len(), nx * ny);
    let mut out = vec![0.0f32; field.len()];
    let idx = |i: usize, j: usize| j * nx + i;

    for j in 0..ny {
        for i in 0..nx {
            let center = field[idx(i, j)];
            let mut laplacian = 0.0f32;

            // X-direction second derivative.
            if i > 0 && i < nx - 1 {
                let left = field[idx(i - 1, j)];
                let right = field[idx(i + 1, j)];
                laplacian += (left - 2.0 * center + right) / (dx * dx);
            } else if i == 0 {
                match bc {
                    BoundaryCondition::Dirichlet => {
                        laplacian += (-2.0 * center) / (dx * dx);
                    }
                    BoundaryCondition::Neumann => {
                        if nx >= 4 {
                            let u1 = field[idx(1, j)];
                            let u2 = field[idx(2, j)];
                            let u3 = field[idx(3, j)];
                            laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) / (dx * dx);
                        } else {
                            let right = field[idx(1, j)];
                            laplacian += (right - 2.0 * center + right) / (dx * dx);
                        }
                    }
                    BoundaryCondition::Periodic => {
                        let left = field[idx(nx - 2, j)];
                        let right = field[idx(1, j)];
                        laplacian += (left - 2.0 * center + right) / (dx * dx);
                    }
                }
            } else {
                // i == nx - 1
                match bc {
                    BoundaryCondition::Dirichlet => {
                        laplacian += (-2.0 * center) / (dx * dx);
                    }
                    BoundaryCondition::Neumann => {
                        if nx >= 4 {
                            let u1 = field[idx(nx - 2, j)];
                            let u2 = field[idx(nx - 3, j)];
                            let u3 = field[idx(nx - 4, j)];
                            laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) / (dx * dx);
                        } else {
                            let left = field[idx(nx - 2, j)];
                            laplacian += (left - 2.0 * center + left) / (dx * dx);
                        }
                    }
                    BoundaryCondition::Periodic => {
                        let left = field[idx(nx - 2, j)];
                        let right = field[idx(1, j)];
                        laplacian += (left - 2.0 * center + right) / (dx * dx);
                    }
                }
            }

            // Y-direction second derivative.
            if j > 0 && j < ny - 1 {
                let bottom = field[idx(i, j - 1)];
                let top = field[idx(i, j + 1)];
                laplacian += (bottom - 2.0 * center + top) / (dy * dy);
            } else if j == 0 {
                match bc {
                    BoundaryCondition::Dirichlet => {
                        laplacian += (-2.0 * center) / (dy * dy);
                    }
                    BoundaryCondition::Neumann => {
                        if ny >= 4 {
                            let u1 = field[idx(i, 1)];
                            let u2 = field[idx(i, 2)];
                            let u3 = field[idx(i, 3)];
                            laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) / (dy * dy);
                        } else {
                            let top = field[idx(i, 1)];
                            laplacian += (top - 2.0 * center + top) / (dy * dy);
                        }
                    }
                    BoundaryCondition::Periodic => {
                        let bottom = field[idx(i, ny - 2)];
                        let top = field[idx(i, 1)];
                        laplacian += (bottom - 2.0 * center + top) / (dy * dy);
                    }
                }
            } else {
                // j == ny - 1
                match bc {
                    BoundaryCondition::Dirichlet => {
                        laplacian += (-2.0 * center) / (dy * dy);
                    }
                    BoundaryCondition::Neumann => {
                        if ny >= 4 {
                            let u1 = field[idx(i, ny - 2)];
                            let u2 = field[idx(i, ny - 3)];
                            let u3 = field[idx(i, ny - 4)];
                            laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) / (dy * dy);
                        } else {
                            let bottom = field[idx(i, ny - 2)];
                            laplacian += (bottom - 2.0 * center + bottom) / (dy * dy);
                        }
                    }
                    BoundaryCondition::Periodic => {
                        let bottom = field[idx(i, ny - 2)];
                        let top = field[idx(i, 1)];
                        laplacian += (bottom - 2.0 * center + top) / (dy * dy);
                    }
                }
            }

            out[idx(i, j)] = laplacian;
        }
    }

    out
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
        nx as u32,
        ny as u32,
        Length::from_unit::<Meter>(dx),
        Length::from_unit::<Meter>(dy),
        bc,
    )
    .unwrap();
    let kernel = Laplacian2DKernel::new(&device).unwrap();
    kernel.dispatch(&device, &input, &output, &params).unwrap();

    let mut got = vec![0.0f32; field.len()];
    device.download(&output, &mut got).unwrap();
    Some(got)
}

#[test]
fn laplacian_minimum_grid_matches_cpu_reference() {
    let field: Vec<f32> = (0..4).map(|i| (i as f32) * 0.5 - 0.75).collect();
    let Some(got) = run_laplacian(&field, 2, 2, 1.0, 1.0, BoundaryCondition::Dirichlet) else {
        return;
    };
    let expected = laplacian_2d_reference(&field, 2, 2, 1.0, 1.0, BoundaryCondition::Dirichlet);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

#[test]
fn laplacian_dirichlet_matches_cpu_reference() {
    let Some(got) = run_laplacian(
        &(0..30).map(|i| i as f32).collect::<Vec<_>>(),
        6,
        5,
        0.5,
        1.0,
        BoundaryCondition::Dirichlet,
    ) else {
        return;
    };
    let expected = laplacian_2d_reference(
        &(0..30).map(|i| i as f32).collect::<Vec<_>>(),
        6,
        5,
        0.5,
        1.0,
        BoundaryCondition::Dirichlet,
    );
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

#[test]
fn laplacian_neumann_matches_cpu_reference() {
    let field: Vec<f32> = (0..30)
        .map(|k| {
            let i = (k % 6) as f32;
            let j = (k / 6) as f32;
            (i + 1.0).ln() + (j + 2.0).sin()
        })
        .collect();

    let Some(got) = run_laplacian(&field, 6, 5, 0.25, 0.75, BoundaryCondition::Neumann) else {
        return;
    };
    let expected = laplacian_2d_reference(&field, 6, 5, 0.25, 0.75, BoundaryCondition::Neumann);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
}

#[test]
fn laplacian_periodic_matches_cpu_reference() {
    let field: Vec<f32> = (0..30)
        .map(|k| {
            let i = (k % 6) as f32;
            let j = (k / 6) as f32;
            i.sin() + j.cos()
        })
        .collect();

    let Some(got) = run_laplacian(&field, 6, 5, 1.0, 1.0, BoundaryCondition::Periodic) else {
        return;
    };
    let expected = laplacian_2d_reference(&field, 6, 5, 1.0, 1.0, BoundaryCondition::Periodic);
    assert_close_slice(&got, &expected, 1e-5, 1e-5);
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
