use crate::infrastructure::device::CudaContext;
use core::marker::PhantomData;
use std::sync::Arc;

use hephaestus_core::DeviceBuffer;

/// A raw CUDA device pointer (`CUdeviceptr`), an opaque device address.
///
/// Kept as cuda-oxide's driver ABI type without exposing cuda-oxide in public
/// APIs; consumers see an opaque integer address for custom kernel launches.
pub type DevicePtr = cuda_oxide::sys::CUdeviceptr;

/// A typed, device-resident linear buffer of `len` elements of `T`.
///
/// The element type lives in `PhantomData<T>` so a buffer allocated for one
/// scalar cannot be passed where another is expected — dtype confusion is a
/// compile error, mirroring [`hephaestus_wgpu::WgpuBuffer`]. The buffer owns
/// its device allocation and frees it on drop.
///
/// [`hephaestus_wgpu::WgpuBuffer`]: https://docs.rs/hephaestus-wgpu
#[derive(Debug)]
pub struct CudaBuffer<T> {
    pub(crate) ptr: DevicePtr,
    pub(crate) len: usize,
    pub(crate) tier: themis::MemoryTier,
    pub(crate) context: Option<Arc<CudaContext>>,
    pub(crate) marker: PhantomData<T>,
}

impl<T> CudaBuffer<T> {
    /// Wrap a raw device pointer and element count.
    ///
    /// `ptr` must be either `0` (the empty-buffer sentinel) or an address
    /// returned by `cuMemAlloc` and not yet freed; the buffer takes ownership
    /// and frees it on drop.
    #[must_use]
    #[inline]
    pub(crate) fn new(
        ptr: DevicePtr,
        len: usize,
        tier: themis::MemoryTier,
        context: Arc<CudaContext>,
    ) -> Self {
        Self {
            ptr,
            len,
            tier,
            context: Some(context),
            marker: PhantomData,
        }
    }

    /// Borrow the raw device pointer for binding into custom kernel launches.
    ///
    /// Consumer escape hatch (parallels `WgpuBuffer::raw`): kernel authors pass
    /// this as a `cuLaunchKernel` parameter over hephaestus-allocated storage.
    #[must_use]
    #[inline]
    pub fn raw(&self) -> DevicePtr {
        self.ptr
    }

    #[must_use]
    #[inline]
    pub(crate) fn aliases<U>(&self, other: &CudaBuffer<U>) -> bool {
        self.ptr != 0 && self.ptr == other.ptr
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

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Some(context) = self.context.take() {
                if context.bind().is_ok() {
                    // SAFETY: `self.ptr` is non-null (guarded above), was
                    // returned by cuda-oxide's `cuMemAlloc_v2` in this context,
                    // and this buffer owns that allocation exactly once.
                    let res = unsafe { cuda_oxide::sys::cuMemFree_v2(self.ptr) };
                    debug_assert_eq!(res, 0, "cuMemFree_v2 failed with code {res}");
                } else {
                    debug_assert!(false, "CudaBuffer drop: context bind failed");
                }
            }
        }
    }
}
