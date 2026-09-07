use super::{HephaestusError, Result, WgpuDevice};

impl WgpuDevice {
    pub(super) fn try_default_with_adapter_config(
        label: &str,
        power_preference: wgpu::PowerPreference,
        select_features: impl Fn(&wgpu::Adapter) -> wgpu::Features,
        select_limits: impl Fn(&wgpu::Adapter) -> wgpu::Limits,
    ) -> Result<Self> {
        // An explicit WGPU_BACKEND selection remains exact. Otherwise only
        // compiled backends participate in acquisition.
        let backends =
            wgpu::Backends::from_env().unwrap_or_else(wgpu::Instance::enabled_backend_features);
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);
        let mut failures = Vec::new();

        let candidates = [
            (
                "hardware",
                wgpu::RequestAdapterOptions {
                    power_preference,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                },
            ),
            (
                "fallback",
                wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                    apply_limit_buckets: false,
                },
            ),
        ];

        for (kind, options) in candidates {
            match futures::executor::block_on(instance.request_adapter(&options)) {
                Err(error) => {
                    failures.push(format!("{backends:?}/{kind}: no adapter ({error})"));
                }
                Ok(adapter) => {
                    // A present adapter's device fault is not recoverable by
                    // choosing another adapter; retain the typed failure.
                    return Self::try_from_adapter_with_features_and_limits(
                        label,
                        &adapter,
                        select_features(&adapter),
                        select_limits(&adapter),
                    );
                }
            }
        }

        Err(HephaestusError::AdapterUnavailable {
            message: format!(
                "no GPU adapter could be acquired for {label:?}; attempts: [{}]",
                failures.join("; ")
            ),
        })
    }
}

#[cfg(test)]
mod tests;
