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
#[derive(Debug)]
pub struct RocmBuffer<T> {
    pub(crate) ptr: DevicePtr,
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
            ptr,
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
        self.ptr
    }

    pub(crate) fn aliases<U>(&self, other: &RocmBuffer<U>) -> bool {
        !self.ptr.is_null() && self.ptr == other.ptr
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
        if self.ptr.is_null() {
            return;
        }
        if self.context.set_current().is_err() {
            debug_assert!(false, "ROCm buffer drop: device selection failed");
            return;
        }
        // SAFETY: `self.ptr` is non-null, was returned by `hipMalloc` for the
        // recorded device, and this buffer owns that allocation exactly once.
        let status = unsafe { cubecl_hip_sys::hipFree(self.ptr) };
        if let Err(error) = check_status(status, "hipFree") {
            debug_assert!(false, "ROCm buffer drop failed: {error}");
        }
    }
}
