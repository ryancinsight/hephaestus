use super::properties::{query_device_features, query_device_limits, query_topology};
use super::{CudaContext, CudaDevice};
use crate::infrastructure::driver::Driver;
use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceFeature,
    DeviceLimits, DevicePreference, HephaestusError, Result,
};
use std::sync::{Arc, Mutex};

// Context creation/binding is serialized; subsequent operations bind per thread.
static CONTEXT_CREATE_LOCK: Mutex<()> = Mutex::new(());

impl CudaDevice {
    /// Acquire the default CUDA device (ordinal 0).
    ///
    /// Returns [`HephaestusError::AdapterUnavailable`] when no CUDA driver or
    /// device is present, rather than fabricating a device. The acquired
    /// device is bound to the calling thread.
    pub fn try_default() -> Result<Self> {
        Self::try_with_ordinal(0)
    }

    /// Acquire a CUDA device by ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::AdapterUnavailable`] when the CUDA driver or
    /// requested ordinal is unavailable. Returns
    /// [`HephaestusError::DeviceUnavailable`] when the context cannot be bound.
    pub fn try_with_ordinal(device_ordinal: usize) -> Result<Self> {
        let device_ordinal =
            i32::try_from(device_ordinal).map_err(|_| HephaestusError::AdapterUnavailable {
                message: format!("CUDA device ordinal {device_ordinal} exceeds i32 range"),
            })?;
        let driver = Driver::get()?;
        let mut device = 0;
        // SAFETY: CUDA is initialized by `Driver::get`; `device` is a valid
        // out-pointer for one driver device handle.
        let status = unsafe { (driver.device.get)(&mut device, device_ordinal) };
        if status != 0 {
            let message = format!("cuDeviceGet({device_ordinal}) -> {status}");
            return Err(if status == 100 || status == 101 {
                HephaestusError::AdapterUnavailable { message }
            } else {
                HephaestusError::DeviceUnavailable { message }
            });
        }
        let context_guard =
            CONTEXT_CREATE_LOCK
                .lock()
                .map_err(|_| HephaestusError::DeviceUnavailable {
                    message: "CUDA context creation lock is poisoned".to_string(),
                })?;
        let context = Arc::new(CudaContext::create(driver, device)?);
        context.bind()?;
        drop(context_guard);
        let limits = query_device_limits(driver, &device)?;
        let features = query_device_features(driver, &device)?;
        let topology = Some(Arc::new(query_topology(driver, &device)?));
        let dev = Self {
            context,
            limits,
            features,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            fusion_pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            topology,
        };
        let buf = dev.alloc_zeroed::<u32>(1)?;
        dev.write_buffer(&buf, &[42u32])?;
        let mut read_val = [0u32];
        dev.download(&buf, &mut read_val)?;
        if read_val != [42] {
            return Err(HephaestusError::DeviceUnavailable {
                message: "CUDA acquisition transfer readback mismatch".to_owned(),
            });
        }

        Ok(dev)
    }

    fn device_count() -> Result<usize> {
        let driver = Driver::get()?;
        let mut count: core::ffi::c_int = 0;
        // SAFETY: `count` is a valid out-pointer for one `c_int`; the CUDA
        // driver has been initialized by `Driver::get`.
        let status = unsafe { (driver.device.count)(&mut count) };
        if status != 0 {
            return Err(HephaestusError::DeviceUnavailable {
                message: format!("CUDA device count query failed with status {status}"),
            });
        }
        usize::try_from(count).map_err(|_| HephaestusError::DeviceUnavailable {
            message: format!("CUDA device count {count} is negative"),
        })
    }

    fn require_limits(actual: DeviceLimits, required: DeviceLimits) -> Result<()> {
        let comparable = [
            (
                "max_buffer_size",
                actual.max_buffer_size,
                required.max_buffer_size,
            ),
            (
                "max_compute_workgroup_size_x",
                u64::from(actual.max_compute_workgroup_size_x),
                u64::from(required.max_compute_workgroup_size_x),
            ),
            (
                "max_compute_workgroup_size_y",
                u64::from(actual.max_compute_workgroup_size_y),
                u64::from(required.max_compute_workgroup_size_y),
            ),
            (
                "max_compute_workgroup_size_z",
                u64::from(actual.max_compute_workgroup_size_z),
                u64::from(required.max_compute_workgroup_size_z),
            ),
            (
                "max_compute_invocations_per_workgroup",
                u64::from(actual.max_compute_invocations_per_workgroup),
                u64::from(required.max_compute_invocations_per_workgroup),
            ),
            (
                "max_compute_workgroup_storage_size",
                u64::from(actual.max_compute_workgroup_storage_size),
                u64::from(required.max_compute_workgroup_storage_size),
            ),
            (
                "max_immediate_size",
                u64::from(actual.max_immediate_size),
                u64::from(required.max_immediate_size),
            ),
        ];
        for (name, available, needed) in comparable {
            if available < needed {
                return Err(HephaestusError::DeviceUnavailable {
                    message: format!(
                        "CUDA device limit {name} {available} is below required {needed}"
                    ),
                });
            }
        }
        if let (Some(available), Some(needed)) = (
            actual.max_storage_buffers_per_shader_stage,
            required.max_storage_buffers_per_shader_stage,
        ) && available < needed
        {
            return Err(HephaestusError::DeviceUnavailable {
                message: format!(
                    "CUDA shader-stage storage-buffer limit {available} is below required {needed}"
                ),
            });
        }
        Ok(())
    }
}

impl ComputeDeviceAcquisition for CudaDevice {
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
