use super::*;

#[test]
fn rejected_device_limit_is_a_fault_without_adapter_fallback() {
    let selections = std::cell::Cell::new(0);
    let requested = std::cell::Cell::new(0);
    let allowed = std::cell::Cell::new(0);
    let result = WgpuDevice::try_with_power_preference_and_adapter_config(
        "rejected-device-limit",
        wgpu::PowerPreference::HighPerformance,
        |_| wgpu::Features::empty(),
        |adapter| {
            selections.set(selections.get() + 1);
            let mut limits = adapter.limits();
            allowed.set(limits.max_texture_dimension_2d);
            limits.max_texture_dimension_2d = limits
                .max_texture_dimension_2d
                .checked_add(1)
                .expect("invariant: a physical texture dimension is below u32::MAX");
            requested.set(limits.max_texture_dimension_2d);
            limits
        },
    );
    match result {
        Err(HephaestusError::DeviceUnavailable { message }) => {
            assert_eq!(
                selections.get(),
                1,
                "a device fault stops adapter selection"
            );
            assert!(message.contains("max_texture_dimension_2d"), "{message}");
            assert!(message.contains(&requested.get().to_string()), "{message}");
            assert!(message.contains(&allowed.get().to_string()), "{message}");
        }
        Err(HephaestusError::AdapterUnavailable { .. })
            if selections.get() == 0
                && std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() => {}
        Err(error) => panic!("device-request failure retains its category: {error}"),
        Ok(_) => panic!("a request beyond the adapter's own limit must fail"),
    }
}
