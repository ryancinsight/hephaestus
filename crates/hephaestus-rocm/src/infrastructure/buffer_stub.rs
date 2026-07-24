use core::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::DeviceBuffer;

use super::DevicePtr;

/// Typed unavailable-path handle used when the ROCm feature is disabled.
#[derive(Debug)]
pub struct RocmBuffer<T> {
    len: usize,
    tier: themis::MemoryTier,
    marker: PhantomData<T>,
}

impl<T: Pod> DeviceBuffer<T> for RocmBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn tier(&self) -> themis::MemoryTier {
        self.tier
    }
}

impl<T> RocmBuffer<T> {
    #[must_use]
    pub(crate) fn raw(&self) -> DevicePtr {
        core::ptr::null_mut()
    }

    pub(crate) fn aliases<U>(&self, _other: &RocmBuffer<U>) -> bool {
        false
    }
}
