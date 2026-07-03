use core::marker::PhantomData;

use hephaestus_core::DeviceBuffer;

/// Stub device buffer for builds without the `cuda` feature.
///
/// No device memory is held; the type exists only to satisfy the
/// [`ComputeDevice::Buffer`] associated type. It is never constructed at
/// runtime because the stub [`CudaDevice`] reports the backend unavailable.
///
/// [`ComputeDevice::Buffer`]: hephaestus_core::ComputeDevice::Buffer
/// [`CudaDevice`]: crate::CudaDevice
#[derive(Debug)]
pub struct CudaBuffer<T> {
    len: usize,
    tier: themis::MemoryTier,
    marker: PhantomData<T>,
}

impl<T> CudaBuffer<T> {
    /// Borrow the raw device pointer (stub returns 0).
    #[must_use]
    #[inline]
    pub fn raw(&self) -> u64 {
        0
    }

    #[must_use]
    #[inline]
    pub(crate) fn aliases<U>(&self, _other: &CudaBuffer<U>) -> bool {
        false
    }
}

impl<T> DeviceBuffer<T> for CudaBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn tier(&self) -> themis::MemoryTier {
        self.tier
    }
}
