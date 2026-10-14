use super::WgpuDevice;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use std::sync::Arc;
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

#[test]
/// Decay evidence: past the idle deadline the parked staging retention is
/// released (the next acquire pays one fresh allocation instead of
/// serving stale parking), and the pool re-warms on subsequent traffic.
fn staging_pool_decays_after_idle_and_rewarm_serves() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-staging-decay") else {
        return;
    };
    let expected = [3_u8; 2048];
    let buffer = device.upload(&expected).expect("upload");
    let accounting = Arc::clone(&device.staging_accounting);
    accounting.test_now_ms.store(1_000, Ordering::Relaxed);

    // Warm the pool: first readback pays the allocation, the repeat is
    // pool-served.
    assert_eq!(
        device.download_owned(&buffer).expect("warm-up download"),
        expected
    );
    assert_eq!(
        device
            .download_owned(&buffer)
            .expect("warm repeat download"),
        expected
    );
    let warm_hits = accounting.hits.load(Ordering::Relaxed);
    let warm_misses = accounting.misses.load(Ordering::Relaxed);
    assert!(warm_hits >= 1, "warm pool must serve a repeat readback");

    // Simulate a session boundary by advancing the injected monotonic
    // clock past the shortened deadline. No wall-clock scheduling enters
    // the state transition or its assertion.
    accounting.test_decay_ms.store(1, Ordering::Relaxed);
    accounting.test_now_ms.store(1_002, Ordering::Relaxed);

    // The acquire inside this readback observes idle-beyond-deadline,
    // clears the parked retention, and pays a fresh allocation.
    assert_eq!(
        device.download_owned(&buffer).expect("post-decay download"),
        expected
    );
    assert_eq!(
        accounting.misses.load(Ordering::Relaxed),
        warm_misses + 1,
        "idle decay must release parked staging retention, forcing one fresh allocation"
    );
    assert_eq!(
        accounting.hits.load(Ordering::Relaxed),
        warm_hits,
        "decayed pool must not serve from stale retention"
    );

    // Restore the production deadline, then confirm the pool re-warms:
    // the buffer parked by the post-decay readback serves the next one.
    accounting.test_decay_ms.store(0, Ordering::Relaxed);
    accounting.test_now_ms.store(0, Ordering::Relaxed);
    assert_eq!(
        device.download_owned(&buffer).expect("rewarm download"),
        expected
    );
    assert_eq!(
        accounting.hits.load(Ordering::Relaxed),
        warm_hits + 1,
        "pool must re-warm and serve after decay"
    );
}
