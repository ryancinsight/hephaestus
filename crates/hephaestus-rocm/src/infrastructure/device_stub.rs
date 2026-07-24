use bytemuck::Pod;
use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceFeature,
    DeviceLimits, DevicePreference, HephaestusError, Result,
};

use super::buffer_stub::RocmBuffer;

fn unavailable() -> HephaestusError {
    HephaestusError::AdapterUnavailable {
        message: "ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
            .to_string(),
    }
}

/// Stub ROCm device used when the optional HIP feature is disabled.
#[derive(Clone, Debug)]
pub struct RocmDevice {
    _unavailable: (),
}

impl RocmDevice {
    /// Report that the optional ROCm backend is unavailable.
    pub fn try_default() -> Result<Self> {
        Err(unavailable())
    }

    /// Report that the optional ROCm backend is unavailable.
    pub fn try_with_ordinal(_device_ordinal: usize) -> Result<Self> {
        Err(unavailable())
    }

    /// No topology exists without an acquired HIP device.
    #[must_use]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        None
    }
}

impl ComputeDeviceCapabilities for RocmDevice {
    fn device_limits(&self) -> DeviceLimits {
        DeviceLimits {
            max_buffer_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroup_storage_size: 0,
            max_storage_buffers_per_shader_stage: None,
            max_buffers_and_acceleration_structures_per_shader_stage: None,
            max_immediate_size: 0,
        }
    }

    fn supports_device_feature(&self, _feature: DeviceFeature) -> bool {
        false
    }
}

impl ComputeDeviceAcquisition for RocmDevice {
    fn try_acquire_device(
        _label: &str,
        _device_preference: DevicePreference,
        _optional_features: &[DeviceFeature],
        _required_limits: DeviceLimits,
    ) -> Result<Self> {
        Err(unavailable())
    }

    fn try_acquire_devices(
        _label_prefix: &str,
        _max_devices: usize,
        _device_preference: DevicePreference,
        _optional_features: &[DeviceFeature],
        _required_limits: DeviceLimits,
    ) -> Result<Vec<Self>> {
        Err(unavailable())
    }
}

impl ComputeDevice for RocmDevice {
    type Buffer<T: Pod> = RocmBuffer<T>;

    fn backend_name(&self) -> &'static str {
        "rocm"
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        _len: usize,
        _hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        Err(unavailable())
    }

    fn upload_with_hint<T: Pod>(
        &self,
        _host: &[T],
        _hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        Err(unavailable())
    }

    fn download<T: Pod>(&self, _buffer: &Self::Buffer<T>, _out: &mut [T]) -> Result<()> {
        Err(unavailable())
    }

    fn write_buffer<T: Pod>(&self, _buffer: &Self::Buffer<T>, _host: &[T]) -> Result<()> {
        Err(unavailable())
    }

    fn write_sub_buffer<T: Pod>(
        &self,
        _buffer: &Self::Buffer<T>,
        _offset: usize,
        _host: &[T],
    ) -> Result<()> {
        Err(unavailable())
    }

    fn synchronize(&self) -> Result<()> {
        Err(unavailable())
    }
}
