use hephaestus_core::{ComputeDeviceCapabilities, DeviceFeature, DeviceLimits};

use super::device::RocmDevice;

impl ComputeDeviceCapabilities for RocmDevice {
    #[inline]
    fn device_limits(&self) -> DeviceLimits {
        self.limits
    }

    #[inline]
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        match feature {
            DeviceFeature::TimestampQuery
            | DeviceFeature::ShaderF64
            | DeviceFeature::ShaderF16
            | DeviceFeature::ImmediateData => false,
            DeviceFeature::MappablePrimaryBuffers => self.features.mappable_primary_buffers,
        }
    }
}
