use core::marker::PhantomData;
use std::sync::Arc;

use hephaestus_core::DeviceBuffer;

use super::{
    DevicePtr,
    device::{RocmContext, check_status},
};

/// A typed, device-resident linear ROCm allocation.
///
/// `T` is carried through `PhantomData` so a buffer's element type remains part
/// of the Rust contract while the HIP runtime sees only an opaque address.
/// The address is stored as an integer, exactly like the CUDA backend: raw
/// pointers are neither `Send` nor `Sync`, and the retained device state of a
/// tracked op must be both. Conversions happen only at this module's FFI
/// boundary, so launch and transfer code keeps receiving `DevicePtr`.
#[derive(Debug)]
pub struct RocmBuffer<T> {
    ptr: u64,
    pub(crate) len: usize,
    pub(crate) tier: themis::MemoryTier,
    pub(crate) context: Arc<RocmContext>,
    marker: PhantomData<T>,
}

impl<T> RocmBuffer<T> {
    pub(crate) fn new(
        ptr: DevicePtr,
        len: usize,
        tier: themis::MemoryTier,
        context: Arc<RocmContext>,
    ) -> Self {
        Self {
            ptr: ptr as u64,
            len,
            tier,
            context,
            marker: PhantomData,
        }
    }

    /// Borrow the opaque HIP device address for a backend kernel launch.
    #[must_use]
    #[inline]
    pub(crate) fn raw(&self) -> DevicePtr {
        self.ptr as DevicePtr
    }

    pub(crate) fn aliases<U>(&self, other: &RocmBuffer<U>) -> bool {
        self.ptr != 0 && self.ptr == other.ptr
    }
}

impl<T> DeviceBuffer<T> for RocmBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn tier(&self) -> themis::MemoryTier {
        self.tier
    }
}

impl<T> Drop for RocmBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        if self.context.set_current().is_err() {
            debug_assert!(false, "ROCm buffer drop: device selection failed");
            return;
        }
        // SAFETY: `self.ptr` is non-zero, was returned by `hipMalloc` for the
        // recorded device, and this buffer owns that allocation exactly once.
        // HIP is 64-bit-only, so the address round-trips through `u64` exactly.
        let status = unsafe { cubecl_hip_sys::hipFree(self.ptr as DevicePtr) };
        if let Err(error) = check_status(status, "hipFree") {
            debug_assert!(false, "ROCm buffer drop failed: {error}");
        }
    }
}
