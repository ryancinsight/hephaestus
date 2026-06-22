use core::marker::PhantomData;

use hephaestus_core::DeviceBuffer;

/// A raw CUDA device pointer (`CUdeviceptr`), an opaque device address.
///
/// Kept as a `u64` (the driver ABI type) so this crate carries no public
/// dependency on cutile-rs types; the `cuda-core` `sys` calls accept it
/// directly, matching coeus-cuda's driver convention.
pub type DevicePtr = u64;

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
    pub(crate) fn new(ptr: DevicePtr, len: usize, tier: themis::MemoryTier) -> Self {
        Self {
            ptr,
            len,
            tier,
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
            // SAFETY: `ptr` is non-zero, so it was returned by allocation functions
            // in `CudaDevice` and has not been freed (drop runs once, and no
            // other `CudaBuffer` holds this address — ownership is unique). The
            // free return code is intentionally ignored: there is no recovery
            // action in a destructor, and a failed free cannot be propagated.
            unsafe {
                if self.tier == themis::MemoryTier::HostPinned {
                    let _ = cuda_core::sys::cuMemFreeHost(self.ptr as *mut core::ffi::c_void);
                } else {
                    let _ = cuda_core::sys::cuMemFree_v2(self.ptr);
                }
            }
        }
    }
}
