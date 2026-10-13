//! hephaestus-core provider trait implementations.

use super::{WgpuDevice, device_wait_deadline, poll_failure};
use crate::infrastructure::buffer::WgpuBuffer;
use core::marker::PhantomData;
use eunomia::Pod;
use hephaestus_core::{
    CommandStream, ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities,
    DeviceFeature, DeviceLimits, DevicePreference, HephaestusError, KernelDevice, Result,
    validate_buffer_size, validate_slice_alignment,
};

impl ComputeDevice for WgpuDevice {
    type Buffer<T: Pod> = WgpuBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "wgpu"
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<WgpuBuffer<T>> {
        // WebGPU guarantees newly created buffers are zero-initialized. Keep
        // the ordinary zeroed contract explicit while sharing the allocation
        // path with callers that overwrite every element before reading.
        self.alloc_uninitialized_with_hint(len, hint)
    }

    fn alloc_uninitialized_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<WgpuBuffer<T>> {
        validate_buffer_size::<T>(len)?;
        let tier = Self::device_tier(hint)?;
        let size = self.storage_size::<T>(len)?;
        let buffer = self.allocate_storage("hephaestus-storage", size, None)?;
        Ok(WgpuBuffer {
            buffer,
            len,
            tier,
            owner: self.pipeline_cache.clone(),
            marker: PhantomData,
        })
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<WgpuBuffer<T>> {
        validate_slice_alignment(host)?;
        let padded_len = self.storage_size::<T>(host.len())?;
        let tier = Self::device_tier(hint)?;
        let payload = Self::padded_host_bytes(host)?;
        let buffer =
            self.allocate_storage("hephaestus-upload", padded_len, Some(payload.as_ref()))?;
        Ok(WgpuBuffer {
            buffer,
            len: host.len(),
            tier,
            owner: self.pipeline_cache.clone(),
            marker: PhantomData,
        })
    }

    fn download<T: Pod>(&self, buffer: &WgpuBuffer<T>, out: &mut [T]) -> Result<()> {
        self.download_into(buffer, out, device_wait_deadline())
    }

    fn download_owned<T: Pod>(&self, buffer: &WgpuBuffer<T>) -> Result<Vec<T>> {
        hephaestus_core::download_owned_into_vec!(T, buffer.len, "WGPU host download", |out| {
            self.download(buffer, out)
        })
    }

    fn write_buffer<T: Pod>(&self, buffer: &WgpuBuffer<T>, host: &[T]) -> Result<()> {
        validate_slice_alignment(host)?;
        if host.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: buffer.len,
            });
        }
        let payload = Self::padded_host_bytes(host)?;
        self.write_bytes(buffer, 0, payload.as_ref())
    }

    #[inline]
    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        WgpuDevice::write_sub_buffer(self, buffer, offset, host)
    }

    fn copy_buffer<T: Pod>(&self, src: &WgpuBuffer<T>, dst: &WgpuBuffer<T>) -> Result<()> {
        let mut stream = self.stream()?;
        stream.copy(src, dst)?;
        // Wait on this copy's own submission index; a whole-queue
        // `synchronize` would also drain every unrelated prior submission.
        let submission_index = stream.submit_indexed()?;
        let deadline = device_wait_deadline();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(deadline),
            })
            .map_err(|error| poll_failure("copy_buffer submission wait", deadline, &error))?;
        Ok(())
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        WgpuDevice::topology(self)
    }

    fn synchronize(&self) -> Result<()> {
        let deadline = device_wait_deadline();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(deadline),
            })
            .map_err(|error| poll_failure("device synchronize", deadline, &error))?;
        Ok(())
    }
}

impl ComputeDeviceCapabilities for WgpuDevice {
    #[inline]
    fn device_limits(&self) -> DeviceLimits {
        WgpuDevice::device_limits(self)
    }

    #[inline]
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        WgpuDevice::supports_device_feature(self, feature)
    }
}

impl ComputeDeviceAcquisition for WgpuDevice {
    fn try_acquire_device(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        Self::try_with_device_preference_and_optional_device_features_and_limits(
            label,
            device_preference,
            optional_features,
            required_limits,
        )
    }

    fn try_acquire_devices(
        label_prefix: &str,
        max_devices: usize,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Vec<Self>> {
        Self::try_enumerate_with_adapter_config_and_rank(
            label_prefix,
            max_devices,
            |info| !matches!(info.backend, wgpu::Backend::BrowserWebGpu),
            |adapter| adapter.features() & Self::wgpu_features(optional_features),
            |_| Self::wgpu_limits_from_device_limits(required_limits),
            |info| Self::adapter_preference_rank(info.device_type, device_preference),
        )
    }
}
