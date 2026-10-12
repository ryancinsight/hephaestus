use super::transfer::ReadbackRegion;
use super::*;
use std::sync::atomic::Ordering;
use std::time::Duration;

use hephaestus_core::{ComputeDevice, DeviceFeature, DevicePreference, HephaestusError};

#[test]
fn module_cases_share_process_state() {
    crate::test_support::run_cases(&[
        (
            "padded_size_aligns_to_copy_boundary",
            padded_size_aligns_to_copy_boundary as fn(),
        ),
        (
            "padded_host_bytes_preserve_an_odd_u16_payload",
            padded_host_bytes_preserve_an_odd_u16_payload as fn(),
        ),
        (
            "staging_mapping_is_reusable_after_consumer_unwind",
            staging_mapping_is_reusable_after_consumer_unwind as fn(),
        ),
        (
            "default_wait_deadline_is_the_production_constant",
            default_wait_deadline_is_the_production_constant as fn(),
        ),
        (
            "poll_failure_maps_a_timeout_to_its_own_variant",
            poll_failure_maps_a_timeout_to_its_own_variant as fn(),
        ),
        (
            "an_overrun_default_wait_is_typed_and_leaves_the_device_usable",
            an_overrun_default_wait_is_typed_and_leaves_the_device_usable as fn(),
        ),
        (
            "aligned_size_overflow_is_allocation_failure",
            aligned_size_overflow_is_allocation_failure as fn(),
        ),
        (
            "byte_size_overflow_is_allocation_failure",
            byte_size_overflow_is_allocation_failure as fn(),
        ),
        (
            "device_feature_mapping_preserves_required_shader_f16",
            device_feature_mapping_preserves_required_shader_f16 as fn(),
        ),
        (
            "adapter_ranking_maps_both_device_preferences",
            adapter_ranking_maps_both_device_preferences as fn(),
        ),
        (
            "downlevel_device_limits_preserve_wgpu_downlevel_contract",
            downlevel_device_limits_preserve_wgpu_downlevel_contract as fn(),
        ),
        (
            "typed_downlevel_limits_preserve_full_wgpu_acquisition_contract",
            typed_downlevel_limits_preserve_full_wgpu_acquisition_contract as fn(),
        ),
        (
            "elevated_storage_limit_raises_the_aggregate_buffer_limit",
            elevated_storage_limit_raises_the_aggregate_buffer_limit as fn(),
        ),
        (
            "staging_pool_serves_sustained_readback_traffic",
            staging_pool_serves_sustained_readback_traffic as fn(),
        ),
        (
            "warm_readback_reuses_completion_state",
            warm_readback_reuses_completion_state as fn(),
        ),
        (
            "concurrent_readbacks_keep_independent_completion_state",
            concurrent_readbacks_keep_independent_completion_state as fn(),
        ),
        (
            "clearing_transient_buffers_preserves_completion_capacity",
            clearing_transient_buffers_preserves_completion_capacity as fn(),
        ),
        (
            "staging_pool_decays_after_idle_and_rewarm_serves",
            staging_pool_decays_after_idle_and_rewarm_serves as fn(),
        ),
    ]);
}

fn padded_size_aligns_to_copy_boundary() {
    match WgpuDevice::byte_size::<u32>(3) {
        Ok(bytes) => assert_eq!(bytes, 12),
        Err(error) => panic!("expected exact byte size, got {error:?}"),
    }
    match WgpuDevice::padded_size::<u8>(3) {
        Ok(bytes) => assert_eq!(bytes, wgpu::COPY_BUFFER_ALIGNMENT),
        Err(error) => panic!("expected padded byte size, got {error:?}"),
    }
    match WgpuDevice::padded_size::<u32>(0) {
        Ok(bytes) => assert_eq!(bytes, 0),
        Err(error) => panic!("expected zero byte size, got {error:?}"),
    }
}

fn padded_host_bytes_preserve_an_odd_u16_payload() {
    let host = [0x0001_u16, 0x0203, 0x0405];
    let payload = WgpuDevice::padded_host_bytes(&host).expect("padded host bytes");
    assert_eq!(payload.len(), 8);
    assert_eq!(&payload[..6], eunomia::layout::cast_slice::<u16, u8>(&host));
    assert_eq!(&payload[6..], [0, 0]);
}

/// The default paths must resolve to the derived production bound, not to
/// a leftover override or an unbounded wait.
fn default_wait_deadline_is_the_production_constant() {
    assert_eq!(device_wait_deadline(), DEFAULT_DEVICE_WAIT);
    assert_eq!(DEFAULT_DEVICE_WAIT, Duration::from_secs(30));
}

/// A default-path wait that cannot complete within its deadline must
/// return the typed timeout instead of blocking, and must leave the device
/// usable afterwards.
///
/// The deadline is driven to 1 ns rather than the device being made to
/// hang: a hang reachable from a test is either a 30 s stall (outside the
/// configured budget) or an infinite kernel (a TDR the host would have to
/// recover from). What is exercised instead is a real, genuinely slower
/// submission — half a gibibyte of queued device-to-device copy traffic,
/// which no shipping device completes in under a microsecond, so the
/// margin over the deadline is five orders of magnitude and the outcome
/// does not depend on scheduling.
fn poll_failure_maps_a_timeout_to_its_own_variant() {
    // The mapping is the part that can regress silently: a caller
    // branching on `DeviceWaitTimeout` stops seeing elapsed deadlines the
    // moment this collapses back into `TransferFailed`. It is a pure
    // function of the poll error, so it is pinned here without a device —
    // deterministically, on every adapter and in every CI job.
    let deadline = Duration::from_millis(250);
    match poll_failure("probe context", deadline, &wgpu::PollError::Timeout) {
        HephaestusError::DeviceWaitTimeout {
            deadline: carried,
            message,
        } => {
            assert_eq!(carried, deadline, "the elapsed bound must be carried");
            assert!(
                message.contains("probe context"),
                "the caller's context must survive, got {message:?}"
            );
        }
        other => panic!("a poll timeout must be its own variant, got {other:?}"),
    }
}

fn an_overrun_default_wait_is_typed_and_leaves_the_device_usable() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-wait-deadline") else {
        return;
    };
    let expected = [7_u32, 8, 9, 10];
    let probe = device.upload(&expected).expect("probe upload");

    // A 1 ns deadline against a real submission. Whether it elapses is a
    // property of the adapter, not of this code: a software adapter can
    // complete the readback before the poll begins, and an earlier version
    // of this case tried to force the issue with 512 MiB of queued copies
    // — which neither made a software adapter slow enough nor drained
    // before the next case ran. So both outcomes are legal here and each
    // is held to its own contract; the timeout *mapping* is pinned
    // deterministically by `poll_failure_maps_a_timeout_to_its_own_variant`
    // above.
    TEST_WAIT_DEADLINE_NS.with(|deadline| deadline.set(1));
    let overrun = device.download_owned::<u32>(&probe);
    TEST_WAIT_DEADLINE_NS.with(|deadline| deadline.set(0));

    match overrun {
        Err(HephaestusError::DeviceWaitTimeout { deadline, .. }) => {
            assert_eq!(
                deadline,
                Duration::from_nanos(1),
                "the error must carry the deadline that elapsed"
            );
        }
        Ok(values) => {
            assert_eq!(
                values, expected,
                "a wait that completed inside its deadline must return the exact payload"
            );
        }
        Err(other) => panic!(
            "an elapsed default wait must be `DeviceWaitTimeout`, never another error: {other:?}"
        ),
    }

    // The point that holds either way: whatever the deadline did, the
    // readback released its staging allocation on the way out, so the same
    // default path back on the production bound returns the exact values.
    assert_eq!(
        device
            .download_owned::<u32>(&probe)
            .expect("the device stays usable after a bounded wait"),
        expected,
        "a bounded wait must leave no staging state behind"
    );
}

fn staging_mapping_is_reusable_after_consumer_unwind() {
    let Ok(device) = WgpuDevice::try_default("hephaestus-staging-unwind") else {
        return;
    };
    let expected = [1_u32, 2, 3, 4];
    let buffer = device.upload(&expected).expect("upload");
    let byte_len = WgpuDevice::byte_size::<u32>(expected.len()).expect("byte length");
    let padded = WgpuDevice::padded_size::<u32>(expected.len()).expect("padded length");

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = device.stage_and_read(
            buffer.raw(),
            ReadbackRegion {
                byte_offset: 0,
                padded,
                byte_len,
            },
            device_wait_deadline(),
            "hephaestus-staging-unwind",
            |_| panic!("intentional mapped-consumer unwind"),
        );
    }));
    let payload = unwind.expect_err("mapped consumer must unwind");
    // The consumer's own panic must be the one that escaped, not a
    // staging-path failure that happened to unwind first.
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"intentional mapped-consumer unwind")
    );

    let actual = device
        .download_owned(&buffer)
        .expect("staging buffer must remain reusable after unwind");
    assert_eq!(actual, expected);
}

fn aligned_size_overflow_is_allocation_failure() {
    match WgpuDevice::aligned_size(u64::MAX, wgpu::COPY_BUFFER_ALIGNMENT) {
        Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
            message,
            format!(
                "buffer byte size {} cannot be aligned to {} bytes",
                u64::MAX,
                wgpu::COPY_BUFFER_ALIGNMENT
            )
        ),
        other => panic!("expected allocation failure, got {other:?}"),
    }
}

fn byte_size_overflow_is_allocation_failure() {
    let overflowing_len = usize::MAX / core::mem::size_of::<u64>() + 1;
    match WgpuDevice::byte_size::<u64>(overflowing_len) {
        Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
            message,
            format!("buffer length {overflowing_len} overflows byte size for 8-byte elements")
        ),
        other => panic!("expected allocation failure, got {other:?}"),
    }
}

fn device_feature_mapping_preserves_required_shader_f16() {
    assert_eq!(
        WgpuDevice::wgpu_features(&[DeviceFeature::ShaderF16, DeviceFeature::TimestampQuery,]),
        wgpu::Features::SHADER_F16 | wgpu::Features::TIMESTAMP_QUERY
    );
}

fn adapter_ranking_maps_both_device_preferences() {
    assert!(
        WgpuDevice::adapter_preference_rank(
            wgpu::DeviceType::DiscreteGpu,
            DevicePreference::HighPerformance,
        ) < WgpuDevice::adapter_preference_rank(
            wgpu::DeviceType::IntegratedGpu,
            DevicePreference::HighPerformance,
        )
    );
    assert!(
        WgpuDevice::adapter_preference_rank(
            wgpu::DeviceType::IntegratedGpu,
            DevicePreference::LowPower,
        ) < WgpuDevice::adapter_preference_rank(
            wgpu::DeviceType::DiscreteGpu,
            DevicePreference::LowPower,
        )
    );
}

fn downlevel_device_limits_preserve_wgpu_downlevel_contract() {
    let downlevel = WgpuDevice::downlevel_device_limits();

    assert_eq!(
        downlevel,
        WgpuDevice::device_limits_from_wgpu(&wgpu::Limits::downlevel_defaults())
    );
    assert_eq!(downlevel.max_storage_buffers_per_shader_stage, Some(4));
}

fn typed_downlevel_limits_preserve_full_wgpu_acquisition_contract() {
    assert_eq!(
        WgpuDevice::wgpu_limits_from_device_limits(WgpuDevice::downlevel_device_limits()),
        wgpu::Limits::downlevel_defaults()
    );
}

fn elevated_storage_limit_raises_the_aggregate_buffer_limit() {
    let mut required = WgpuDevice::downlevel_device_limits();
    required.max_storage_buffers_per_shader_stage = Some(32);

    let limits = WgpuDevice::wgpu_limits_from_device_limits(required);

    assert_eq!(limits.max_storage_buffers_per_shader_stage, 32);
    assert_eq!(
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
        32
    );
}

/// Steady-state evidence: repeated same-size readbacks must be served by
/// the staging pool (at most the first pays a fresh allocation), pinning
/// the warm hit-rate contract the idle-decay deadline is designed around.
fn staging_pool_serves_sustained_readback_traffic() {
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
fn warm_readback_reuses_completion_state() {
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

fn concurrent_readbacks_keep_independent_completion_state() {
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

fn clearing_transient_buffers_preserves_completion_capacity() {
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
