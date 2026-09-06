use super::CudaDevice;
use hephaestus_core::{ComputeDeviceCapabilities, DeviceFeature, DeviceLimits};

impl ComputeDeviceCapabilities for CudaDevice {
    #[inline]
    fn device_limits(&self) -> DeviceLimits {
        self.limits
    }

    #[inline]
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        match feature {
            DeviceFeature::TimestampQuery
            | DeviceFeature::ShaderF16
            | DeviceFeature::MappablePrimaryBuffers => false,
            DeviceFeature::ShaderF64 => self.features.shader_f64,
            DeviceFeature::ImmediateData => self.features.immediate_data,
        }
    }
}
