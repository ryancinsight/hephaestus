use hephaestus_core::{
    ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceFeature, DeviceLimits,
    DevicePreference, Result,
};

use super::device::RocmDevice;

impl ComputeDeviceAcquisition for RocmDevice {
    fn try_acquire_device(
        _label: &str,
        _device_preference: DevicePreference,
        _optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        let device = Self::try_default()?;
        Self::require_limits(device.device_limits(), required_limits)?;
        Ok(device)
    }

    fn try_acquire_devices(
        _label_prefix: &str,
        max_devices: usize,
        _device_preference: DevicePreference,
        _optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Vec<Self>> {
        let count = Self::device_count()?;
        let mut devices = Vec::with_capacity(count.min(max_devices));
        for ordinal in 0..count.min(max_devices) {
            let device = Self::try_with_ordinal(ordinal)?;
            Self::require_limits(device.device_limits(), required_limits)?;
            devices.push(device);
        }
        Ok(devices)
    }
}
