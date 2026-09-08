use super::WgpuDevice;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use std::sync::atomic::Ordering;

fn acquire() -> Result<Option<WgpuDevice>> {
    match WgpuDevice::try_default("transient pool ownership") {
        Ok(device) => Ok(Some(device)),
        Err(HephaestusError::AdapterUnavailable { .. })
            if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[test]
fn transfer_pools_reject_enabled_limit_overflow() -> Result<()> {
    let Some(probe) = acquire()? else {
        return Ok(());
    };
    let mut limits = probe.limits();
    limits.max_buffer_size = 256;
    limits.max_storage_buffer_binding_size = 256;
    limits.max_uniform_buffer_binding_size = 256;
    let device = WgpuDevice::try_default_with_limits("transfer pool limits", limits)?;
    for (outcome, expected) in [
        (device.get_staging_buffer(257), "264"),
        (device.get_uniform_buffer(257), "260"),
    ] {
        match outcome {
            Err(HephaestusError::AllocationFailed { message }) => {
                assert!(message.contains(expected), "{message}");
                assert!(message.contains("max_buffer_size=256"), "{message}");
            }
            other => panic!("expected enabled-limit rejection, got {other:?}"),
        }
    }
    let expected = [19_u8; 256];
    let buffer = device.upload(&expected)?;
    assert_eq!(device.download_owned(&buffer)?, expected);
    Ok(())
}

#[test]
fn transient_owners_return_to_their_creating_pool() -> Result<()> {
    let Some(first) = acquire()? else {
        return Ok(());
    };
    let second = WgpuDevice::try_default("second transient pool owner")?;
    let staging = [first.get_staging_buffer(8)?, second.get_staging_buffer(8)?];
    let uniforms = [first.get_uniform_buffer(4)?, second.get_uniform_buffer(12)?];
    // Owners cross host scopes in reverse order; recycling has no device
    // argument that could accidentally select the other device's pool.
    for owner in staging.into_iter().rev() {
        drop(owner);
    }
    for owner in uniforms.into_iter().rev() {
        drop(owner);
    }
    for (device, expected_size) in [(&first, 4), (&second, 12)] {
        let returned = device
            .uniform_pool
            .take_at_least(expected_size)
            .expect("invariant: the uniform owner returns its allocation to its creating pool");
        assert_eq!(returned.size(), expected_size);
        assert_eq!(
            returned.usage(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        );
    }
    for (device, expected) in [(&first, [7_u32, 11]), (&second, [31, 47])] {
        let source = device.upload(&expected)?;
        let mut actual = [0; 2];
        device.download(&source, &mut actual)?;
        assert_eq!(actual, expected);
        assert_eq!(device.staging_accounting.hits.load(Ordering::Relaxed), 1);
        assert_eq!(device.staging_accounting.misses.load(Ordering::Relaxed), 1);
    }
    Ok(())
}
