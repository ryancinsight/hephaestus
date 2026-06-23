use core::marker::PhantomData;
use hephaestus_core::DeviceBuffer;

/// A typed device buffer over a `wgpu::Buffer`.
///
/// The element type lives in `PhantomData<T>` so dtype confusion between
/// buffers is a compile error; the raw `wgpu::Buffer` carries no type. The
/// element count is stored explicitly because the underlying allocation is
/// padded to wgpu's copy alignment and may exceed `len * size_of::<T>()`.
///
/// # Cloning
///
/// `Clone` is a GPU handle clone (similar to cloning an `Arc`): both the
/// original and the clone refer to the **same** device allocation. Use
/// [`aliases`](WgpuBuffer::aliases) to detect aliased pairs before
/// dispatching kernels with output aliasing.
#[derive(Clone, Debug)]
pub struct WgpuBuffer<T> {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) len: usize,
    pub(crate) tier: themis::MemoryTier,
    pub(crate) marker: PhantomData<T>,
}

impl<T> WgpuBuffer<T> {
    /// Construct a `WgpuBuffer` wrapper from a raw buffer and element count.
    ///
    /// # Safety
    ///
    /// The caller must ensure `len` equals the number of `T`-elements that fit
    /// in the allocation (i.e. `len * size_of::<T>() <= buffer.size()`). This
    /// function is `pub(crate)` because only the `ComputeDevice` impl provides
    /// validated construction paths.
    #[must_use]
    #[inline]
    pub(crate) fn new(buffer: wgpu::Buffer, len: usize, tier: themis::MemoryTier) -> Self {
        debug_assert!(
            len.checked_mul(core::mem::size_of::<T>())
                .map_or(false, |bytes| bytes <= buffer.size() as usize),
            "invariant: len * size_of::<T>() must fit within the wgpu buffer allocation"
        );
        Self {
            buffer,
            len,
            tier,
            marker: PhantomData,
        }
    }

    /// Borrow the raw `wgpu::Buffer` for binding into custom pipelines.
    ///
    /// This is the consumer escape hatch: apollo's transform kernels build
    /// their own bind groups over hephaestus-allocated storage.
    #[must_use]
    #[inline]
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    #[must_use]
    #[inline]
    pub(crate) fn aliases<U>(&self, other: &WgpuBuffer<U>) -> bool {
        self.buffer == other.buffer
    }
}

impl<T> std::ops::Deref for WgpuBuffer<T> {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T> DeviceBuffer<T> for WgpuBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn tier(&self) -> themis::MemoryTier {
        self.tier
    }
}
