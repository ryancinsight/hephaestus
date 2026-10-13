//! Native-precision cross-entropy backward capability contract.
//! Supported precision executes the shared Leto oracle; unsupported precision
//! rejects the operation before mutating any provider-resident operand.
//!
//! Forward stays out of this binary: the f64 forward shaders need `exp` and
//! `log`, whose f64 SPIR-V lowering is undefined, so the backend refuses them
//! on Vulkan (ADR 0072) and only the backward path (pure IEEE-double
//! arithmetic) executes here. CUDA proves
//! the f64 forward path on hardware.

use hephaestus_conformance::assert_cross_entropy_backward_contract_for;
use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyOps, DeviceFeature, DevicePreference,
    HephaestusError, StridedView,
};
use hephaestus_wgpu::{WgpuCrossEntropyOps, WgpuDevice};
use leto::Layout;

fn device() -> Option<WgpuDevice> {
    // The f64 arithmetic preflight binds five storage buffers plus the
    // uniform buffer, above the downlevel four.
    let mut limits = WgpuDevice::downlevel_device_limits();
    limits.max_storage_buffers_per_shader_stage = Some(5);
    limits.max_buffers_and_acceleration_structures_per_shader_stage = Some(6);
    match WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "hephaestus-cross-entropy-precision-contract",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF64],
        limits,
    ) {
        Ok(device) => Some(device),
        Err(HephaestusError::AdapterUnavailable { message })
            if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none()
                && std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_none() =>
        {
            eprintln!("cross-entropy adapter unavailable: {message}");
            None
        }
        Err(error) => panic!("cross-entropy precision contract requires a device: {error}"),
    }
}

#[test]
fn backward_respects_precision_capabilities() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let device = {
        let _phase = tracing::info_span!("arrange", contract = "precision_capability").entered();
        let Some(device) = device() else { return };
        device
    };
    if !device.supports_device_feature(DeviceFeature::ShaderF64) {
        let probabilities_values = [0.25_f64, 0.75];
        let gradient_values = [3.0_f64, 5.0];
        let probabilities = device.upload(&probabilities_values).expect("probabilities");
        let gradient = device.upload(&gradient_values).expect("gradient");
        let upstream = device.upload(&[2.0_f64]).expect("upstream");
        let targets = device.upload(&[1_u32]).expect("targets");
        let scalar_layout = Layout::c_contiguous([1]).expect("scalar layout");
        let matrix_layout = Layout::c_contiguous([1, 2]).expect("matrix layout");
        let error = {
            let _phase = tracing::info_span!("act", contract = "precision_rejection").entered();
            WgpuCrossEntropyOps
                .cross_entropy_backward_accumulate(
                    &device,
                    CrossEntropyBackwardOperands {
                        output_gradient: StridedView::new(&upstream, &scalar_layout),
                        probabilities: StridedView::new(&probabilities, &matrix_layout),
                        targets: StridedView::new(&targets, &scalar_layout),
                        logit_gradient: StridedView::new(&gradient, &matrix_layout),
                    },
                )
                .expect_err("unsupported native precision must reject cross-entropy")
        };
        let _phase = tracing::info_span!("assert", contract = "no_mutation").entered();
        match error {
            HephaestusError::InvalidConfiguration { message } => assert_eq!(
                message,
                "WGPU cross-entropy requires the ShaderF64 device feature for f64"
            ),
            other => panic!("unexpected precision rejection: {other}"),
        }
        assert_eq!(
            device.download_owned(&gradient).expect("gradient readback"),
            gradient_values
        );
        assert_eq!(
            device
                .download_owned(&probabilities)
                .expect("probabilities readback"),
            probabilities_values
        );
        assert_eq!(
            device.download_owned(&upstream).expect("upstream readback"),
            [2.0]
        );
        assert_eq!(
            device.download_owned(&targets).expect("target readback"),
            [1]
        );
        return;
    }
    let _phase = tracing::info_span!("act", contract = "native_value_oracle").entered();
    assert_cross_entropy_backward_contract_for::<_, _, f64>(&device, &WgpuCrossEntropyOps);
}
