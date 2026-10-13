//! Callback delivery, deadline, and wakeup contracts for retained and overflow slots.

use super::*;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn callback_publication_delivers_both_outcomes_across_slot_kinds() {
    let _trace = trace();
    for capacity in [0, 1] {
        for outcome in [MAP_SUCCEEDED, MAP_FAILED] {
            let (pool, reader, callback) = {
                let _arrange = tracing::info_span!("arrange", capacity, outcome).entered();
                let pool = Arc::new(MapCompletionPool::new(capacity));
                let (reader, callback) = pool.acquire();
                (pool, reader, callback)
            };
            let start = std::sync::Barrier::new(2);
            let result = {
                let _act = tracing::info_span!("act").entered();
                std::thread::scope(|scope| {
                    let waiter = scope.spawn(|| {
                        start.wait();
                        reader.wait(Instant::now(), Duration::from_secs(1), "publication")
                    });
                    start.wait();
                    callback.complete(if outcome == MAP_SUCCEEDED {
                        Ok(())
                    } else {
                        Err(wgpu::BufferAsyncError)
                    });
                    waiter.join().expect("completion waiter")
                })
            };
            let _assert = tracing::info_span!("assert").entered();
            assert_eq!(reader.slot().state.load(Ordering::Acquire), outcome);
            if outcome == MAP_SUCCEEDED {
                result.expect("published successful mapping");
            } else {
                assert!(matches!(
                    result,
                    Err(HephaestusError::TransferFailed { message })
                        if message == "buffer mapping failed"
                ));
            }
            drop(reader);
            assert_eq!(pool.retained_in_flight_count(), 0);
            assert_eq!(pool.overflow_allocation_count(), u64::from(capacity == 0));
        }
    }
}

#[test]
fn callback_timeout_quarantines_late_publication_and_preserves_reuse() {
    let _trace = trace();
    let (pool, reader, callback, timeout) = {
        let _arrange = tracing::info_span!("arrange").entered();
        let pool = Arc::new(MapCompletionPool::new(1));
        let (reader, callback) = pool.acquire();
        (pool, reader, callback, Duration::from_millis(2))
    };
    let started = Instant::now();
    let result = {
        let _act = tracing::info_span!("act").entered();
        reader.wait(started, timeout, "delayed-readback")
    };
    let _assert = tracing::info_span!("assert").entered();
    assert!(matches!(
        result,
        Err(HephaestusError::DeviceWaitTimeout { deadline, message })
            if deadline == timeout && message == "delayed-readback: map_async callback delivery"
    ));
    assert!(started.elapsed() >= timeout);
    assert_eq!(reader.slot().state.load(Ordering::Acquire), MAP_PENDING);
    drop(reader);
    assert_eq!(pool.retained_in_flight_count(), 1);

    let (overflow_reader, overflow_callback) = pool.acquire();
    assert_eq!(pool.overflow_allocation_count(), 1);
    callback.complete(Ok(()));
    assert_eq!(pool.retained_in_flight_count(), 0);
    overflow_callback.complete(Err(wgpu::BufferAsyncError));
    assert!(matches!(
        overflow_reader.wait(Instant::now(), Duration::ZERO, "overflow"),
        Err(HephaestusError::TransferFailed { message }) if message == "buffer mapping failed"
    ));

    let (reused_reader, reused_callback) = pool.acquire();
    assert_eq!(pool.overflow_allocation_count(), 1);
    assert_eq!(
        reused_reader.slot().state.load(Ordering::Acquire),
        MAP_PENDING
    );
    reused_callback.complete(Ok(()));
    reused_reader
        .wait(Instant::now(), Duration::ZERO, "reused")
        .expect("reused successful mapping");
    assert_eq!(
        reused_reader.slot().state.load(Ordering::Acquire),
        MAP_SUCCEEDED
    );
    drop(reused_reader);
    assert_eq!(pool.retained_in_flight_count(), 0);
}

#[test]
fn callback_wait_does_not_restart_an_expired_submission_budget() {
    let _trace = trace();
    let (reader, callback, timeout, started) = {
        let _arrange = tracing::info_span!("arrange").entered();
        let pool = Arc::new(MapCompletionPool::new(1));
        let (reader, callback) = pool.acquire();
        let timeout = Duration::from_millis(2);
        let started = Instant::now().checked_sub(timeout).expect("elapsed budget");
        (reader, callback, timeout, started)
    };
    let result = {
        let _act = tracing::info_span!("act").entered();
        reader.wait(started, timeout, "submission")
    };
    let _assert = tracing::info_span!("assert").entered();
    assert!(matches!(
        result,
        Err(HephaestusError::DeviceWaitTimeout { deadline, message })
            if deadline == timeout && message == "submission: map_async callback delivery"
    ));
    assert_eq!(reader.slot().state.load(Ordering::Acquire), MAP_PENDING);
    callback.complete(Ok(()));
    reader
        .wait(started, timeout, "submission")
        .expect("already delivered callback remains observable at the deadline");
    assert_eq!(reader.slot().state.load(Ordering::Acquire), MAP_SUCCEEDED);
}

/// Enumerate publication before/after the reader's predicate check, notification
/// before/after parking, and a wakeup that does not change the predicate.
#[test]
fn guarded_publication_has_no_lost_wakeup_under_all_interleavings() {
    let _trace = trace();
    let _act = tracing::info_span!("act").entered();
    loom::model(|| {
        let state = loom::sync::Arc::new(loom::sync::atomic::AtomicU8::new(MAP_PENDING));
        let delivery =
            loom::sync::Arc::new((loom::sync::Mutex::new(()), loom::sync::Condvar::new()));
        let reader_state = state.clone();
        let reader_delivery = delivery.clone();
        let reader = loom::thread::spawn(move || {
            let (lock, ready) = &*reader_delivery;
            let mut guard = lock.lock().expect("reader guard");
            while reader_state.load(Ordering::Acquire) == MAP_PENDING {
                guard = ready.wait(guard).expect("reader wakeup");
            }
            assert_eq!(reader_state.load(Ordering::Acquire), MAP_SUCCEEDED);
        });
        let callback_state = state.clone();
        let callback_delivery = delivery.clone();
        let callback = loom::thread::spawn(move || {
            let (lock, ready) = &*callback_delivery;
            let _guard = lock.lock().expect("callback guard");
            callback_state.store(MAP_SUCCEEDED, Ordering::Release);
            ready.notify_one();
        });
        let spurious = loom::thread::spawn(move || delivery.1.notify_one());
        reader.join().expect("reader terminates after publication");
        callback.join().expect("callback terminates");
        spurious.join().expect("independent wakeup");
        assert_eq!(state.load(Ordering::Acquire), MAP_SUCCEEDED);
    });
}
