//! Feature and limit mapping for device acquisition.

use super::WgpuDevice;
use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference};

impl WgpuDevice {
    #[inline]
    pub(super) const fn wgpu_power_preference(
        preference: DevicePreference,
    ) -> wgpu::PowerPreference {
        match preference {
            DevicePreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            DevicePreference::LowPower => wgpu::PowerPreference::LowPower,
        }
    }

    pub(super) const fn adapter_preference_rank(
        device_type: wgpu::DeviceType,
        preference: DevicePreference,
    ) -> u8 {
        match (preference, device_type) {
            (DevicePreference::HighPerformance, wgpu::DeviceType::DiscreteGpu)
            | (DevicePreference::LowPower, wgpu::DeviceType::IntegratedGpu) => 0,
            (DevicePreference::HighPerformance, wgpu::DeviceType::IntegratedGpu)
            | (DevicePreference::LowPower, wgpu::DeviceType::DiscreteGpu) => 1,
            (_, wgpu::DeviceType::VirtualGpu) => 2,
            (_, wgpu::DeviceType::Other) => 3,
            (_, wgpu::DeviceType::Cpu) => 4,
        }
    }

    #[inline]
    pub(super) const fn wgpu_feature(feature: DeviceFeature) -> wgpu::Features {
        match feature {
            DeviceFeature::TimestampQuery => wgpu::Features::TIMESTAMP_QUERY,
            DeviceFeature::ShaderF64 => wgpu::Features::SHADER_F64,
            DeviceFeature::ShaderF16 => wgpu::Features::SHADER_F16,
            DeviceFeature::MappablePrimaryBuffers => wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
            DeviceFeature::ImmediateData => wgpu::Features::IMMEDIATES,
        }
    }

    pub(super) fn wgpu_features(features: &[DeviceFeature]) -> wgpu::Features {
        features
            .iter()
            .copied()
            .fold(wgpu::Features::empty(), |acc, feature| {
                acc | Self::wgpu_feature(feature)
            })
    }

    #[inline]
    pub(super) const fn device_limits_from_wgpu(limits: &wgpu::Limits) -> DeviceLimits {
        DeviceLimits {
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_storage_buffers_per_shader_stage: Some(limits.max_storage_buffers_per_shader_stage),
            max_buffers_and_acceleration_structures_per_shader_stage: Some(
                limits.max_buffers_and_acceleration_structures_per_shader_stage,
            ),
            max_immediate_size: limits.max_immediate_size,
        }
    }

    pub(super) fn wgpu_limits_from_device_limits(required: DeviceLimits) -> wgpu::Limits {
        let downlevel = wgpu::Limits::downlevel_defaults();
        let max_storage_buffers = required
            .max_storage_buffers_per_shader_stage
            .unwrap_or(downlevel.max_storage_buffers_per_shader_stage);
        // Prefer the consumer's explicit combined budget when supplied — only it
        // can account for uniform buffers bound alongside storage buffers. Fall
        // back to deriving from the storage-buffer request for callers that pin
        // only the legacy limit. Storage buffers always count against the
        // aggregate, so it can never fall below the storage request.
        let max_buffers_and_acceleration_structures = required
            .max_buffers_and_acceleration_structures_per_shader_stage
            .unwrap_or(downlevel.max_buffers_and_acceleration_structures_per_shader_stage)
            .max(max_storage_buffers);

        wgpu::Limits {
            max_buffer_size: required.max_buffer_size,
            max_compute_workgroup_size_x: required.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: required.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: required.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: required.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: required.max_compute_workgroup_storage_size,
            max_buffers_and_acceleration_structures_per_shader_stage:
                max_buffers_and_acceleration_structures,
            max_storage_buffers_per_shader_stage: max_storage_buffers,
            max_immediate_size: required.max_immediate_size,
            ..downlevel
        }
    }

    /// WGPU backend default limits mapped into the backend-neutral Hephaestus vocabulary.
    #[must_use]
    pub fn default_device_limits() -> DeviceLimits {
        Self::device_limits_from_wgpu(&wgpu::Limits::default())
    }

    /// WGPU downlevel limits mapped into the backend-neutral Hephaestus vocabulary.
    ///
    /// Use this acquisition baseline when a consumer must retain compatibility
    /// with WGPU downlevel adapters while raising only the limits its kernels
    /// require.
    #[must_use]
    pub fn downlevel_device_limits() -> DeviceLimits {
        Self::device_limits_from_wgpu(&wgpu::Limits::downlevel_defaults())
    }
}
