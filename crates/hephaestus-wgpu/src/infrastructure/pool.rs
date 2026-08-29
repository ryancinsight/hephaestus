//! Transient buffer and readback-completion pooling.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use hephaestus_core::{HephaestusError, Result};

use crate::infrastructure::device::WgpuDevice;

const MAP_PENDING: u8 = 0;
const MAP_SUCCEEDED: u8 = 1;
const MAP_FAILED: u8 = 2;

#[derive(Debug)]
struct MapCompletionSlot {
    state: AtomicU8,
    retained_claimed: AtomicBool,
    retained_users: AtomicU8,
}

impl MapCompletionSlot {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(MAP_PENDING),
            retained_claimed: AtomicBool::new(false),
            retained_users: AtomicU8::new(0),
        }
    }
}

/// One owner of completion state for an asynchronous WGPU buffer mapping.
///
/// The pool creates exactly two owners per mapping: the synchronous reader and
/// the WGPU callback. A retained slot stays claimed until both owners drop, so
/// a delayed callback can never write into state reused by another mapping.
#[derive(Debug)]
pub(crate) struct MapCompletion {
    storage: MapCompletionStorage,
}

#[derive(Debug)]
enum MapCompletionStorage {
    Retained {
        pool: Arc<MapCompletionPool>,
        index: usize,
    },
    Overflow(Arc<MapCompletionSlot>),
}

impl MapCompletion {
    fn retained(pool: Arc<MapCompletionPool>, index: usize) -> Self {
        Self {
            storage: MapCompletionStorage::Retained { pool, index },
        }
    }

    fn overflow(slot: Arc<MapCompletionSlot>) -> Self {
        Self {
            storage: MapCompletionStorage::Overflow(slot),
        }
    }

    fn slot(&self) -> &MapCompletionSlot {
        match &self.storage {
            MapCompletionStorage::Retained { pool, index } => &pool.slots[*index],
            MapCompletionStorage::Overflow(slot) => slot,
        }
    }

    pub(crate) fn complete(self, result: core::result::Result<(), wgpu::BufferAsyncError>) {
        self.finish(if result.is_ok() {
            MAP_SUCCEEDED
        } else {
            MAP_FAILED
        });
    }

    fn finish(self, state: u8) {
        // The release publishes the callback result to the reader's acquire.
        self.slot().state.store(state, Ordering::Release);
    }

    pub(crate) fn result(&self) -> Result<()> {
        // The acquire observes all callback work sequenced before completion.
        match self.slot().state.load(Ordering::Acquire) {
            MAP_SUCCEEDED => Ok(()),
            MAP_FAILED => Err(HephaestusError::TransferFailed {
                message: "buffer mapping failed".to_owned(),
            }),
            _ => Err(HephaestusError::TransferFailed {
                message: "map_async callback was not delivered after device completion".to_owned(),
            }),
        }
    }
}

impl Drop for MapCompletion {
    fn drop(&mut self) {
        let MapCompletionStorage::Retained { pool, index } = &self.storage else {
            return;
        };
        let slot = &pool.slots[*index];
        if slot.retained_users.fetch_sub(1, Ordering::AcqRel) == 1 {
            // The release prevents a later acquirer from resetting the slot
            // before both the reader and callback have finished with it.
            slot.retained_claimed.store(false, Ordering::Release);
        }
    }
}

/// Bounded retained completion slots for synchronous readback callbacks.
///
/// Each acquisition claims one fixed slot and returns independent reader and
/// callback owners. Concurrency above the retained capacity allocates an
/// unpooled overflow slot before any GPU submission rather than serializing
/// device work. Pending callbacks keep their slot quarantined until WGPU drops
/// or invokes them.
#[derive(Debug)]
pub(crate) struct MapCompletionPool {
    slots: Box<[MapCompletionSlot]>,
    #[cfg(test)]
    overflow_allocations: std::sync::atomic::AtomicU64,
}

impl MapCompletionPool {
    pub(crate) fn new(retained_capacity: usize) -> Self {
        let slots = (0..retained_capacity)
            .map(|_| MapCompletionSlot::new())
            .collect();
        Self {
            slots,
            #[cfg(test)]
            overflow_allocations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Acquire reader and callback owners for one mapping.
    ///
    /// This must run before queue submission so capacity overflow cannot make
    /// an already-dispatched transfer fallible due to a host allocation.
    pub(crate) fn acquire(self: &Arc<Self>) -> (MapCompletion, MapCompletion) {
        for (index, slot) in self.slots.iter().enumerate() {
            // The acquire pairs with the prior last owner's release. Setting
            // `true` grants this thread exclusive initialization of the slot.
            if slot
                .retained_claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                slot.state.store(MAP_PENDING, Ordering::Relaxed);
                slot.retained_users.store(2, Ordering::Relaxed);
                return (
                    MapCompletion::retained(Arc::clone(self), index),
                    MapCompletion::retained(Arc::clone(self), index),
                );
            }
        }

        #[cfg(test)]
        self.overflow_allocations.fetch_add(1, Ordering::Relaxed);
        let slot = Arc::new(MapCompletionSlot::new());
        (
            MapCompletion::overflow(Arc::clone(&slot)),
            MapCompletion::overflow(slot),
        )
    }

    #[cfg(test)]
    fn retained_in_flight_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.retained_claimed.load(Ordering::Acquire))
            .count()
    }

    #[cfg(test)]
    pub(crate) fn overflow_allocation_count(&self) -> u64 {
        self.overflow_allocations.load(Ordering::Relaxed)
    }
}

/// Zero-cost orphan-rule wrapper around wgpu::Buffer that implements SizeBounded.
#[repr(transparent)]
pub struct PoolBuffer(pub wgpu::Buffer);

impl moirai_sync::SizeBounded for PoolBuffer {
    #[inline]
    fn size(&self) -> u64 {
        self.0.size()
    }
}

impl std::ops::Deref for PoolBuffer {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Generic RAII guard that recycles a wgpu buffer back to a pool on drop.
///
/// The recycle strategy `F` is a function that returns the buffer to its pool.
/// Callers use the type-aliased guards [`StagingBufferGuard`] and
/// [`UniformBufferGuard`] rather than instantiating this type directly.
///
/// This is the SSOT for all pooled-buffer RAII logic; both guard variants share
/// the identical fields, `Deref` impl, and `Drop` impl.
pub struct PoolBufferGuard<F>
where
    F: Fn(&WgpuDevice, wgpu::Buffer),
{
    device: WgpuDevice,
    buffer: Option<wgpu::Buffer>,
    recycle: F,
}

impl<F: Fn(&WgpuDevice, wgpu::Buffer)> PoolBufferGuard<F> {
    #[inline]
    #[must_use]
    pub(crate) fn new(device: WgpuDevice, buffer: wgpu::Buffer, recycle: F) -> Self {
        Self {
            device,
            buffer: Some(buffer),
            recycle,
        }
    }
}

impl<F: Fn(&WgpuDevice, wgpu::Buffer)> std::ops::Deref for PoolBufferGuard<F> {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("invariant: buffer is not dropped")
    }
}

impl<F: Fn(&WgpuDevice, wgpu::Buffer)> Drop for PoolBufferGuard<F> {
    #[inline]
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            (self.recycle)(&self.device, buffer);
        }
    }
}

/// RAII guard that automatically recycles a staging buffer back to the device's pool on drop.
pub type StagingBufferGuard = PoolBufferGuard<fn(&WgpuDevice, wgpu::Buffer)>;

/// RAII guard that automatically recycles a uniform buffer back to the device's pool on drop.
pub type UniformBufferGuard = PoolBufferGuard<fn(&WgpuDevice, wgpu::Buffer)>;

/// Construct a [`StagingBufferGuard`] — wraps a buffer that is returned to the
/// staging pool on drop.
#[inline]
#[must_use]
pub(crate) fn staging_guard(device: WgpuDevice, buffer: wgpu::Buffer) -> StagingBufferGuard {
    PoolBufferGuard::new(device, buffer, |d, b| d.recycle_staging_buffer(b))
}

/// Construct a [`UniformBufferGuard`] — wraps a buffer that is returned to the
/// uniform pool on drop.
#[inline]
#[must_use]
pub(crate) fn uniform_guard(device: WgpuDevice, buffer: wgpu::Buffer) -> UniformBufferGuard {
    PoolBufferGuard::new(device, buffer, |d, b| d.recycle_uniform_buffer(b))
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    struct ModelSlot {
        state: loom::sync::atomic::AtomicU8,
        claimed: loom::sync::atomic::AtomicBool,
        users: loom::sync::atomic::AtomicU8,
    }

    impl ModelSlot {
        fn acquired() -> Self {
            Self {
                state: loom::sync::atomic::AtomicU8::new(MAP_PENDING),
                claimed: loom::sync::atomic::AtomicBool::new(true),
                users: loom::sync::atomic::AtomicU8::new(2),
            }
        }

        fn release(&self) {
            if self.users.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.claimed.store(false, Ordering::Release);
            }
        }

        fn try_reacquire(&self, callback_finishes: bool) -> bool {
            match self
                .claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(false) => {
                    assert_eq!(
                        self.users.load(Ordering::Acquire),
                        0,
                        "a retained slot cannot be reacquired while either prior owner remains"
                    );
                    if callback_finishes {
                        assert_eq!(
                            self.state.load(Ordering::Acquire),
                            MAP_SUCCEEDED,
                            "callback completion must be published before the slot is released"
                        );
                    }
                    true
                }
                Err(true) => false,
                result => panic!("invalid one-slot claim transition: {result:?}"),
            }
        }
    }

    fn model_two_owner_reclamation(callback_finishes: bool) {
        loom::model(move || {
            let slot = loom::sync::Arc::new(ModelSlot::acquired());

            assert_eq!(
                slot.claimed
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed,),
                Err(true),
                "capacity-plus-one must overflow while both retained owners remain"
            );

            let reader_slot = loom::sync::Arc::clone(&slot);
            let reader = loom::thread::spawn(move || reader_slot.release());

            let callback_slot = loom::sync::Arc::clone(&slot);
            let callback = loom::thread::spawn(move || {
                if callback_finishes {
                    callback_slot.state.store(MAP_SUCCEEDED, Ordering::Release);
                }
                callback_slot.release();
            });

            let acquirer_slot = loom::sync::Arc::clone(&slot);
            let acquirer =
                loom::thread::spawn(move || acquirer_slot.try_reacquire(callback_finishes));

            reader.join().expect("reader owner");
            callback.join().expect("callback owner");
            let raced_reacquisition = acquirer.join().expect("racing acquirer");
            if !raced_reacquisition {
                assert!(
                    slot.try_reacquire(callback_finishes),
                    "slot must be reusable after both prior owners terminate"
                );
            }
        });
    }

    #[test]
    fn two_owner_reclamation_is_generation_safe_under_all_interleavings() {
        model_two_owner_reclamation(true);
        model_two_owner_reclamation(false);
    }

    #[test]
    fn held_capacity_uses_preallocated_slots_then_overflows() {
        let pool = Arc::new(MapCompletionPool::new(2));
        let (first_reader, first_callback) = pool.acquire();
        let (second_reader, second_callback) = pool.acquire();

        assert_eq!(pool.retained_in_flight_count(), 2);
        assert_eq!(pool.overflow_allocation_count(), 0);

        let (overflow_reader, overflow_callback) = pool.acquire();
        assert_eq!(pool.retained_in_flight_count(), 2);
        assert_eq!(pool.overflow_allocation_count(), 1);

        first_callback.finish(MAP_SUCCEEDED);
        second_callback.finish(MAP_FAILED);
        overflow_callback.finish(MAP_SUCCEEDED);
        match first_reader.result() {
            Ok(()) => {}
            Err(error) => panic!("successful completion reported {error:?}"),
        }
        assert!(matches!(
            second_reader.result(),
            Err(HephaestusError::TransferFailed { message })
                if message == "buffer mapping failed"
        ));
        match overflow_reader.result() {
            Ok(()) => {}
            Err(error) => panic!("successful overflow completion reported {error:?}"),
        }

        assert_eq!(pool.retained_in_flight_count(), 2);
        drop(first_reader);
        assert_eq!(pool.retained_in_flight_count(), 1);
        drop(second_reader);
        assert_eq!(pool.retained_in_flight_count(), 0);
    }

    #[test]
    fn pending_callback_quarantines_slot_until_callback_termination() {
        let pool = Arc::new(MapCompletionPool::new(1));
        let (reader, callback) = pool.acquire();
        assert_eq!(pool.retained_in_flight_count(), 1);
        assert!(matches!(
            reader.result(),
            Err(HephaestusError::TransferFailed { message })
                if message == "map_async callback was not delivered after device completion"
        ));

        drop(reader);
        assert_eq!(pool.retained_in_flight_count(), 1);

        let (overflow_reader, overflow_callback) = pool.acquire();
        assert_eq!(pool.overflow_allocation_count(), 1);
        drop(overflow_reader);
        drop(overflow_callback);

        callback.finish(MAP_SUCCEEDED);
        assert_eq!(pool.retained_in_flight_count(), 0);

        let (reused_reader, reused_callback) = pool.acquire();
        assert_eq!(pool.overflow_allocation_count(), 1);
        reused_callback.finish(MAP_SUCCEEDED);
        match reused_reader.result() {
            Ok(()) => {}
            Err(error) => panic!("reused completion reported {error:?}"),
        }
        drop(reused_reader);
        assert_eq!(pool.retained_in_flight_count(), 0);

        let (cancelled_reader, cancelled_callback) = pool.acquire();
        drop(cancelled_reader);
        assert_eq!(pool.retained_in_flight_count(), 1);
        drop(cancelled_callback);
        assert_eq!(pool.retained_in_flight_count(), 0);
    }
}
