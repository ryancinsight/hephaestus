//! Native Metal acquisition contracts for the canonical WGPU provider (ADR 0047).
//!
//! Successful acquisition retains Metal identity and computes every element of
//! `a + b` exactly in unsigned integer arithmetic. A 257-element operand crosses
//! the default 256-lane workgroup boundary. Enumeration returns at most its
//! requested bound; an impossible device capacity rejects without another API.

use hephaestus_core::{
    ComputeDevice, DeviceFeature, DeviceLimits, DevicePreference, HephaestusError,
};
use hephaestus_wgpu::{AddOp, WgpuDevice, binary_elementwise, wgpu};

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_metal("hephaestus-native-metal-contracts") {
        Ok(device) => Some(device),
        Err(HephaestusError::AdapterUnavailable { message })
            if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_none() =>
        {
            tracing::info!(%message, "Metal adapter unavailable");
            None
        }
        Err(error) => panic!("native Metal acquisition must succeed: {error}"),
    }
}

fn assert_native_values(device: &WgpuDevice) {
    let (left, right, expected) = {
        let _phase = tracing::info_span!("arrange", contract = "native_values").entered();
        let left: Vec<_> = (0..257_u32).collect();
        let right: Vec<_> = left.iter().map(|value| 3 * value + 1).collect();
        // For i in [0, 257), i + (3i + 1) = 4i + 1 <= 1025 < 2^32.
        let expected: Vec<_> = left.iter().map(|value| 4 * value + 1).collect();
        (
            device.upload(&left).expect("left upload"),
            device.upload(&right).expect("right upload"),
            expected,
        )
    };
    let actual = {
        let _phase = tracing::info_span!("act", contract = "native_values").entered();
        let result =
            binary_elementwise::<AddOp, _>(device, &left, &right).expect("native Metal dispatch");
        device.download_owned(&result).expect("readback")
    };
    let _phase = tracing::info_span!("assert", contract = "native_identity_and_values").entered();
    assert_eq!(device.backend_name(), "wgpu");
    assert_eq!(
        device.adapter_info().expect("adapter identity").backend,
        wgpu::Backend::Metal
    );
    assert_eq!(actual, expected);
}

#[test]
fn metal_acquisition_preserves_identity_features_limits_and_values() {
    let _trace = trace();
    let reference = {
        let _phase = tracing::info_span!("arrange").entered();
        let Some(device) = device_or_skip() else {
            return;
        };
        device
    };
    assert_native_values(&reference);
    let required_limits = reference.device_limits();
    let optional_features = [
        DeviceFeature::TimestampQuery,
        DeviceFeature::ShaderF16,
        DeviceFeature::ShaderF64,
    ];
    let requested_features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16 | wgpu::Features::SHADER_F64;

    for preference in [
        DevicePreference::HighPerformance,
        DevicePreference::LowPower,
    ] {
        let device = {
            let _phase = tracing::info_span!("act", ?preference).entered();
            WgpuDevice::try_metal_with_device_preference_and_optional_device_features_and_limits(
                "hephaestus-native-metal-preference",
                preference,
                &optional_features,
                required_limits,
            )
            .expect("available Metal adapter satisfies its enabled limits")
        };
        let _phase = tracing::info_span!("assert", ?preference).entered();
        assert_eq!(device.device_limits(), required_limits);
        assert_eq!(
            device.features(),
            device.adapter_features().expect("adapter capabilities") & requested_features
        );
        assert_native_values(&device);
    }

    let devices = {
        let _phase = tracing::info_span!("act", contract = "bounded_enumeration").entered();
        WgpuDevice::try_enumerate_metal_with_optional_device_features_and_limits(
            "hephaestus-native-metal-enumeration",
            1,
            DevicePreference::HighPerformance,
            &optional_features,
            required_limits,
        )
        .expect("Metal enumeration")
    };
    let _phase = tracing::info_span!("assert", contract = "bounded_enumeration").entered();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].device_limits(), required_limits);
    assert_native_values(&devices[0]);
}

#[test]
fn metal_zero_enumeration_bound_returns_no_devices() {
    let _trace = trace();
    let limits = {
        let _phase = tracing::info_span!("arrange").entered();
        DeviceLimits {
            max_buffer_size: u64::MAX,
            max_compute_workgroup_size_x: 1,
            max_compute_workgroup_size_y: 1,
            max_compute_workgroup_size_z: 1,
            max_compute_invocations_per_workgroup: 1,
            max_compute_workgroup_storage_size: 1,
            max_storage_buffers_per_shader_stage: Some(1),
            max_buffers_and_acceleration_structures_per_shader_stage: Some(1),
            max_immediate_size: 0,
        }
    };
    let devices = {
        let _phase = tracing::info_span!("act").entered();
        WgpuDevice::try_enumerate_metal_with_optional_device_features_and_limits(
            "hephaestus-native-metal-empty-enumeration",
            0,
            DevicePreference::HighPerformance,
            &[],
            limits,
        )
        .expect("zero bound must not acquire a device")
    };
    let _phase = tracing::info_span!("assert").entered();
    assert!(devices.is_empty(), "zero bound returned device handles");
}

#[test]
fn metal_impossible_limits_reject_without_backend_fallback() {
    let _trace = trace();
    let limits = {
        let _phase = tracing::info_span!("arrange").entered();
        let Some(device) = device_or_skip() else {
            return;
        };
        let mut limits = device.device_limits();
        // No physical allocation can span the entire unsigned address range.
        limits.max_buffer_size = u64::MAX;
        limits
    };
    let outcome = {
        let _phase = tracing::info_span!("act").entered();
        WgpuDevice::try_metal_with_device_preference_and_optional_device_features_and_limits(
            "hephaestus-native-metal-impossible-capacity",
            DevicePreference::HighPerformance,
            &[],
            limits,
        )
    };
    let _phase = tracing::info_span!("assert").entered();
    match outcome {
        Err(HephaestusError::AdapterUnavailable { message }) => assert_eq!(
            message,
            "No compatible Metal GPU adapter or device could be acquired."
        ),
        other => panic!("impossible Metal limits must reject acquisition: {other:?}"),
    }
}
