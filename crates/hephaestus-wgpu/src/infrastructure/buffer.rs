use core::marker::PhantomData;
use hephaestus_core::DeviceBuffer;

/// A typed device buffer over a `wgpu::Buffer`.
///
/// The element type lives in `PhantomData<T>` so dtype confusion between
/// buffers is a compile error; the raw `wgpu::Buffer` carries no type. The
/// element count is stored explicitly because the underlying allocation is
/// padded to wgpu's copy alignment and may exceed `len * size_of::<T>()`.
#[derive(Debug)]
pub struct WgpuBuffer<T> {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) len: usize,
    pub(crate) marker: PhantomData<T>,
}

impl<T> WgpuBuffer<T> {
    /// Borrow the raw `wgpu::Buffer` for binding into custom pipelines.
    ///
    /// This is the consumer escape hatch: apollo's transform kernels build
    /// their own bind groups over hephaestus-allocated storage.
    #[must_use]
    #[inline]
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<T> DeviceBuffer<T> for WgpuBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}
