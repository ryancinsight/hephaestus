//! Acquire a real contract device without conflating optional precision with
//! required adapter availability. Missing optional features stay disabled;
//! their operation contracts must assert rejection before output mutation.

use hephaestus_core::{DeviceFeature, DevicePreference};
use hephaestus_wgpu::WgpuDevice;

pub(super) fn device_with_optional_features(
    label: &str,
    features: &[DeviceFeature],
) -> Option<WgpuDevice> {
    let reference = super::device_or_skip()?;
    let limits = reference.device_limits();
    let acquired = if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
        WgpuDevice::try_metal_with_device_preference_and_optional_device_features_and_limits(
            label,
            DevicePreference::HighPerformance,
            features,
            limits,
        )
    } else {
        WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
            label,
            DevicePreference::HighPerformance,
            features,
            limits,
        )
    };
    Some(acquired.expect("available adapter satisfies its enabled limits and optional features"))
}
