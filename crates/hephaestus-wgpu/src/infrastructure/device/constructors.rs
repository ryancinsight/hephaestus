//! Fallible device-acquisition constructors.

use super::WgpuDevice;
use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference, HephaestusError, Result};
use std::sync::Arc;

impl WgpuDevice {
    /// Acquire a default high-performance adapter and device.
    ///
    /// `label` becomes the wgpu device label. Uses
    /// [`wgpu::Limits::downlevel_defaults`]; for custom limits use
    /// [`try_default_with_limits`](Self::try_default_with_limits).
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    #[inline]
    pub fn try_default(label: &str) -> Result<Self> {
        Self::try_default_with_limits(label, wgpu::Limits::downlevel_defaults())
    }

    /// Acquire a default adapter and device with custom limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    #[inline]
    pub fn try_default_with_limits(label: &str, required_limits: wgpu::Limits) -> Result<Self> {
        Self::try_default_with_features_and_limits(label, wgpu::Features::empty(), required_limits)
    }

    /// Acquire a default adapter and device with custom features and limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_default_with_features_and_limits(
        label: &str,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            required_limits,
            wgpu::PowerPreference::HighPerformance,
            |_| required_features,
        )
    }

    /// Acquire a default adapter and device, enabling optional features only
    /// when the selected adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_default_with_optional_features_and_limits(
        label: &str,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            required_limits,
            wgpu::PowerPreference::HighPerformance,
            |adapter| adapter.features() & optional_features,
        )
    }

    /// Acquire an adapter matching `device_preference`, enabling optional
    /// features only when the selected adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_optional_features_and_limits(
        label: &str,
        device_preference: DevicePreference,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_with_power_preference_and_optional_features_and_limits(
            label,
            Self::wgpu_power_preference(device_preference),
            optional_features,
            required_limits,
        )
    }

    /// Acquire an adapter matching `device_preference`, enabling optional
    /// Hephaestus features only when the selected adapter reports support for
    /// them. Uses the backend's default WGPU limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_optional_device_features(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
    ) -> Result<Self> {
        Self::try_with_device_preference_and_optional_features_and_limits(
            label,
            device_preference,
            Self::wgpu_features(optional_features),
            wgpu::Limits::default(),
        )
    }

    /// Acquire an adapter matching `device_preference` and require every
    /// requested Hephaestus device feature.
    ///
    /// Uses the backend's downlevel-default WGPU limits. Adapter selection
    /// returns a device failure when the selected adapter cannot create a device
    /// with the complete requested feature set; required features never drop.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter can be acquired.
    /// [`HephaestusError::DeviceUnavailable`] when the selected adapter cannot
    /// create a device with every requested feature.
    pub fn try_with_device_preference_and_required_device_features(
        label: &str,
        device_preference: DevicePreference,
        required_features: &[DeviceFeature],
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            wgpu::Limits::downlevel_defaults(),
            Self::wgpu_power_preference(device_preference),
            |_| Self::wgpu_features(required_features),
        )
    }

    /// Acquire an adapter matching `device_preference`, enabling optional
    /// Hephaestus features when supported and applying backend-neutral required
    /// compute limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_optional_device_features_and_limits(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        Self::try_with_device_preference_and_optional_features_and_limits(
            label,
            device_preference,
            Self::wgpu_features(optional_features),
            Self::wgpu_limits_from_device_limits(required_limits),
        )
    }

    /// Acquire an adapter matching `power_preference`, enabling optional
    /// features only when the selected adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_power_preference_and_optional_features_and_limits(
        label: &str,
        power_preference: wgpu::PowerPreference,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_features_and_limits(
            label,
            required_limits,
            power_preference,
            |adapter| adapter.features() & optional_features,
        )
    }

    /// Acquire an adapter matching `device_preference`, deriving both required
    /// features and required limits from the selected adapter.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_device_preference_and_adapter_config(
        label: &str,
        device_preference: DevicePreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Self> {
        Self::try_with_power_preference_and_adapter_config(
            label,
            Self::wgpu_power_preference(device_preference),
            select_features,
            select_limits,
        )
    }

    /// Acquire an adapter matching `power_preference`, deriving both required
    /// features and required limits from the selected adapter.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no adapter exists on this
    /// host; [`HephaestusError::DeviceUnavailable`] when device creation fails.
    pub fn try_with_power_preference_and_adapter_config(
        label: &str,
        power_preference: wgpu::PowerPreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Self> {
        Self::try_default_with_adapter_config(
            label,
            power_preference,
            select_features,
            select_limits,
        )
    }

    /// Enumerate adapters and create devices for those accepted by
    /// `accept_adapter`, deriving each device descriptor from the selected
    /// adapter.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for any accepted adapter.
    pub fn try_enumerate_with_adapter_config(
        label_prefix: &str,
        max_devices: usize,
        accept_adapter: impl Fn(&wgpu::AdapterInfo) -> bool,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Vec<Self>> {
        Self::try_enumerate_with_adapter_config_and_rank(
            label_prefix,
            max_devices,
            accept_adapter,
            select_features,
            select_limits,
            |_| 0,
        )
    }

    pub(super) fn try_enumerate_with_adapter_config_and_rank(
        label_prefix: &str,
        max_devices: usize,
        accept_adapter: impl Fn(&wgpu::AdapterInfo) -> bool,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
        rank_adapter: impl Fn(&wgpu::AdapterInfo) -> u8,
    ) -> Result<Vec<Self>> {
        if max_devices == 0 {
            return Ok(Vec::new());
        }

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Instance::enabled_backend_features();
        let instance = wgpu::Instance::new(desc);
        let mut devices = Vec::new();
        let mut adapters = futures::executor::block_on(
            instance.enumerate_adapters(wgpu::Instance::enabled_backend_features()),
        );
        adapters.retain(|adapter| accept_adapter(&adapter.get_info()));
        adapters.sort_by_key(|adapter| rank_adapter(&adapter.get_info()));

        for adapter in adapters {
            let info = adapter.get_info();
            let label = format!("{label_prefix}: {}", info.name);
            let required_features = select_features(&adapter);
            let required_limits = select_limits(&adapter);
            if !required_limits.check_limits(&adapter.limits()) {
                continue;
            }
            devices.push(Self::try_from_adapter_with_features_and_limits(
                &label,
                &adapter,
                required_features,
                required_limits,
            )?);

            if devices.len() >= max_devices {
                break;
            }
        }

        Ok(devices)
    }

    /// Acquire one Metal-backed adapter matching `device_preference`, enabling
    /// optional features only when supported and enforcing backend-neutral
    /// compute limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no Metal adapter can satisfy
    /// the request; [`HephaestusError::DeviceUnavailable`] when logical-device
    /// creation fails.
    pub fn try_metal_with_device_preference_and_optional_device_features_and_limits(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        Self::try_enumerate_metal_with_optional_device_features_and_limits(
            label,
            1,
            device_preference,
            optional_features,
            required_limits,
        )
        .and_then(|mut devices| {
            devices
                .pop()
                .ok_or_else(|| HephaestusError::AdapterUnavailable {
                    message: "No compatible Metal GPU adapter or device could be acquired."
                        .to_string(),
                })
        })
    }

    /// Acquire up to `max_devices` Metal-backed devices with optional features
    /// and backend-neutral required limits.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for an enumerated Metal adapter.
    pub fn try_enumerate_metal_with_optional_device_features_and_limits(
        label_prefix: &str,
        max_devices: usize,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Vec<Self>> {
        Self::try_enumerate_with_adapter_config_and_rank(
            label_prefix,
            max_devices,
            |info| matches!(info.backend, wgpu::Backend::Metal),
            |adapter| adapter.features() & Self::wgpu_features(optional_features),
            |_| Self::wgpu_limits_from_device_limits(required_limits),
            |info| Self::adapter_preference_rank(info.device_type, device_preference),
        )
    }

    /// Create a device from a caller-selected adapter, enabling optional
    /// features only when that adapter reports support for them.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for the supplied adapter.
    pub fn try_from_adapter_with_optional_features_and_limits(
        label: &str,
        adapter: &wgpu::Adapter,
        optional_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        let required_features = adapter.features() & optional_features;
        Self::try_from_adapter_with_features_and_limits(
            label,
            adapter,
            required_features,
            required_limits,
        )
    }

    /// Create a device from a caller-selected adapter with exact required
    /// features.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when logical-device creation
    /// fails for the supplied adapter.
    pub fn try_from_adapter_with_features_and_limits(
        label: &str,
        adapter: &wgpu::Adapter,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self> {
        let (device, queue) =
            futures::executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                required_features,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            }))
            .map_err(|error| HephaestusError::DeviceUnavailable {
                message: error.to_string(),
            })?;
        Ok(Self::new(Arc::new(device), Arc::new(queue)).with_adapter_metadata(adapter))
    }

    fn try_default_with_adapter_features_and_limits(
        label: &str,
        required_limits: wgpu::Limits,
        power_preference: wgpu::PowerPreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
    ) -> Result<Self> {
        Self::try_default_with_adapter_config(label, power_preference, select_features, move |_| {
            required_limits.clone()
        })
    }

    /// Acquire a default adapter and device, specifically targeting the Metal backend.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::AdapterUnavailable`] when no Metal adapter can be acquired.
    pub fn try_metal(label: &str) -> Result<Self> {
        Self::try_metal_with_device_preference_and_optional_device_features_and_limits(
            label,
            DevicePreference::HighPerformance,
            &[],
            Self::downlevel_device_limits(),
        )
    }
}
