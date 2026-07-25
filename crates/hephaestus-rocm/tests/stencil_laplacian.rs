//! ROCm differential contracts for the 2D Laplacian stencil.

use aequitas::systems::si::{quantities::Length, units::Meter};
use hephaestus_core::{
    BoundaryCondition, ComputeDevice, HephaestusError, Laplacian2DParams, LaplacianPolarity,
};
use hephaestus_rocm::{Laplacian2DKernel, RocmDevice};
use leto::{Array1, Laplacian2D};
use leto_ops::laplacian_2d_into;

fn device(test: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_some() {
                panic!("ROCm device required for {test}: {error}");
            }
            eprintln!("skipping ROCm stencil contract {test}: {error}");
            None
        }
    }
}

fn field() -> Vec<f32> {
    (0..30)
        .map(|k| {
            let i = (k % 6) as f32;
            let j = (k / 6) as f32;
            (i + 1.0).ln() + (j + 2.0).sin()
        })
        .collect()
}

fn reference(input: &[f32], boundary: BoundaryCondition, polarity: LaplacianPolarity) -> Vec<f32> {
    let input = Array1::from_shape_vec([input.len()], input.to_vec()).expect("input shape");
    let mut output = Array1::zeros([input.len()]);
    let stencil = Laplacian2D::new(
        6,
        5,
        Length::from_unit::<Meter>(0.25),
        Length::from_unit::<Meter>(0.75),
        boundary,
    )
    .expect("valid reference grid")
    .with_polarity(polarity);
    laplacian_2d_into(&stencil, &input.view(), &mut output.view_mut()).expect("reference storage");
    output.iter().copied().collect()
}

#[test]
fn boundaries_and_polarities_match_cpu_reference() {
    let Some(device) = device("boundaries_and_polarities_match_cpu_reference") else {
        return;
    };
    let input = field();
    let input_buffer = device.upload(&input).expect("input upload");
    let kernel = Laplacian2DKernel::new(&device).expect("kernel compile");
    for boundary in [
        BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann,
        BoundaryCondition::Periodic,
    ] {
        for polarity in [
            LaplacianPolarity::Laplacian,
            LaplacianPolarity::NegativeLaplacian,
        ] {
            let output = device
                .alloc_zeroed::<f32>(input.len())
                .expect("output allocation");
            let params = Laplacian2DParams::new(
                6,
                5,
                Length::from_unit::<Meter>(0.25),
                Length::from_unit::<Meter>(0.75),
                boundary,
                polarity,
            )
            .expect("valid parameters");
            kernel
                .dispatch(&device, &input_buffer, &output, &params)
                .expect("stencil dispatch");
            let mut actual = vec![0.0f32; input.len()];
            device
                .download(&output, &mut actual)
                .expect("output download");
            let expected = reference(&input, boundary, polarity);
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "mismatch at {index}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn storage_length_mismatch_is_rejected_before_launch() {
    let Some(device) = device("storage_length_mismatch_is_rejected_before_launch") else {
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
