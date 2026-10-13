//! Staging-pool and readback tests.

use super::*;

/// Steady-state evidence: repeated same-size readbacks must be served by
/// the staging pool (at most the first pays a fresh allocation), pinning
/// the warm hit-rate contract the idle-decay deadline is designed around.
pub(super) fn staging_pool_serves_sustained_readback_traffic() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-staging-hit-rate") else {
        return;
    };
    let expected = [7_u8; 4096];
    let buffer = device.upload(&expected).expect("upload");
    let accounting = Arc::clone(&device.staging_accounting);

    let rounds: u64 = 16;
    for round in 0..rounds as usize {
        let actual = device
            .download_owned(&buffer)
            .unwrap_or_else(|error| panic!("readback {round} failed: {error:?}"));
        assert_eq!(actual, expected);
    }

    let hits = accounting.hits.load(Ordering::Relaxed);
    let misses = accounting.misses.load(Ordering::Relaxed);
    assert!(
        misses <= 1 && hits + misses >= rounds,
        "steady-state readbacks must be pool-served (hits={hits}, misses={misses})"
    );
}

/// Lifecycle evidence: device construction owns the bounded completion
/// slots, so neither the first nor a repeated readback allocates one.
pub(super) fn warm_readback_reuses_completion_state() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-map-completion-reuse") else {
        return;
    };
    let expected = [11_u32, 13, 17, 19];
    let buffer = device.upload(&expected).expect("upload");
    let mut actual = [0_u32; 4];
    let prepared_allocations = device.map_completion_pool.overflow_allocation_count();

    device
        .download(&buffer, &mut actual)
        .expect("warm readback");
    assert_eq!(actual, expected);
    let warm_allocations = device.map_completion_pool.overflow_allocation_count();
    assert_eq!(
        warm_allocations, prepared_allocations,
        "first readback must use device-owned callback completion state"
    );

    actual.fill(0);
    device
        .download(&buffer, &mut actual)
        .expect("reused readback");
    assert_eq!(actual, expected);
    assert_eq!(
        device.map_completion_pool.overflow_allocation_count(),
        warm_allocations,
        "warm readback must reuse retained callback completion state"
    );
}

pub(super) fn concurrent_readbacks_keep_independent_completion_state() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-map-completion-concurrency") else {
        return;
    };
    let first_expected = [2_u32, 3, 5, 7];
    let second_expected = [23_u32, 29, 31, 37];
    let first = device.upload(&first_expected).expect("first upload");
    let second = device.upload(&second_expected).expect("second upload");
    let barrier = std::sync::Barrier::new(3);
    let prepared_allocations = device.map_completion_pool.overflow_allocation_count();

    std::thread::scope(|scope| {
        let first_download = scope.spawn(|| {
            barrier.wait();
            let mut actual = [0_u32; 4];
            device
                .download(&first, &mut actual)
                .expect("first concurrent readback");
            actual
        });
        let second_download = scope.spawn(|| {
            barrier.wait();
            let mut actual = [0_u32; 4];
            device
                .download(&second, &mut actual)
                .expect("second concurrent readback");
            actual
        });
        barrier.wait();

        assert_eq!(
            first_download.join().expect("first readback thread"),
            first_expected
        );
        assert_eq!(
            second_download.join().expect("second readback thread"),
            second_expected
        );
    });
    assert_eq!(
        device.map_completion_pool.overflow_allocation_count(),
        prepared_allocations,
        "two concurrent readbacks must use distinct device-owned slots"
    );
}

pub(super) fn clearing_transient_buffers_preserves_completion_capacity() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-map-completion-clear") else {
        return;
    };
    let expected = [41_u32, 43, 47, 53];
    let buffer = device.upload(&expected).expect("upload");
    let prepared_allocations = device.map_completion_pool.overflow_allocation_count();

    device.clear_transient_pools();
    let mut actual = [0_u32; 4];
    device
        .download(&buffer, &mut actual)
        .expect("readback after transient-buffer clear");

    assert_eq!(actual, expected);
    assert_eq!(
        device.map_completion_pool.overflow_allocation_count(),
        prepared_allocations,
        "clearing transient buffers must preserve fixed completion capacity"
    );
}

/// Decay evidence: past the idle deadline the parked staging retention is
/// released (the next acquire pays one fresh allocation instead of
/// serving stale parking), and the pool re-warms on subsequent traffic.
pub(super) fn staging_pool_decays_after_idle_and_rewarm_serves() {
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
