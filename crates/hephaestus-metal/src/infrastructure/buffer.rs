use hephaestus_core::DeviceBuffer;
use hephaestus_wgpu::WgpuBuffer;

/// A typed, device-resident Metal linear buffer of `len` elements of `T`.
#[derive(Clone, Debug)]
pub struct MetalBuffer<T> {
    pub(crate) inner: WgpuBuffer<T>,
}

impl<T> MetalBuffer<T> {
    /// Borrow the underlying WgpuBuffer context.
    #[must_use]
    #[inline]
    pub fn wgpu_buffer(&self) -> &WgpuBuffer<T> {
        &self.inner
    }
}

impl<T> DeviceBuffer<T> for MetalBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn tier(&self) -> themis::MemoryTier {
        self.inner.tier()
    }
}
