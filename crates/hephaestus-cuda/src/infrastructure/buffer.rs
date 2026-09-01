use crate::infrastructure::device::{CudaContext, CurrentContext};
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
    /// `ptr` must be either `0` (the zero-byte allocation sentinel, including
    /// zero-sized element buffers) or an address returned by `cuMemAlloc` and
    /// not yet freed; the buffer takes ownership and frees it on drop.
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
        if self.ptr != 0
            && let Some(context) = self.context.take()
        {
            // Freeing requires the owning context current, but this drop can
            // run between a consumer's `device.bind()` and a raw-handle
            // driver call (`cuLaunchKernel` over `raw`); restore the
            // previously current context so the drop-time bind cannot
            // silently retarget the dropping thread.
            let previous = CurrentContext::capture();
            if context.bind().is_ok() {
                // Drop soundness. A buffer may still be referenced by a kernel
                // in flight when it is dropped, so the free must not take
                // effect before that kernel completes.
                //
                // `cuMemFree_v2` gave that by synchronizing the entire device —
                // correct, and the reason it was safe, but it drained all
                // in-flight work on every drop. `cuMemFreeAsync` on the null
                // stream gives the same guarantee by ordering instead of
                // blocking: the free is enqueued behind the work already
                // submitted to that stream, and this backend launches every
                // kernel and issues every copy on that same null stream. The
                // ordering is contractual rather than incidental, which is what
                // `HEPH-CUDA-STREAM-ORDERED-ALLOC` required before async frees
                // could replace the synchronizing pair.
                //
                // Allocation and free read the same per-device flag, so they
                // cannot diverge. That pairing is a performance contract, not a
                // validity one: the driver does accept `cuMemFree_v2` on a
                // stream-ordered allocation (measured — the suite passes with
                // the branch forced), but it reintroduces the device-wide
                // synchronization this change exists to remove.
                let res = if context.stream_ordered {
                    // SAFETY: `self.ptr` is non-null (guarded above), was
                    // returned by `cuMemAllocAsync` on the null stream in this
                    // context, and this buffer owns that allocation exactly
                    // once. Freeing on the same stream orders the release
                    // after every prior use of the pointer on it.
                    unsafe { cuda_oxide::sys::cuMemFreeAsync(self.ptr, std::ptr::null_mut()) }
                } else {
                    // SAFETY: `self.ptr` is non-null (guarded above), was
                    // returned by cuda-oxide's `cuMemAlloc_v2` in this context,
                    // and this buffer owns that allocation exactly once.
                    unsafe { cuda_oxide::sys::cuMemFree_v2(self.ptr) }
                };
                debug_assert_eq!(res, 0, "device free failed with code {res}");
            } else {
                debug_assert!(false, "CudaBuffer drop: context bind failed");
            }
            if let Some(previous) = previous {
                previous.restore();
            }
        }
    }
}
