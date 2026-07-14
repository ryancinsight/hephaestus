//! Page-locked ("pinned") host memory (CU-P6/CU-M3).
//!
//! Every host<->device transfer in this crate previously staged through a
//! plain `Vec<T>` (pageable memory): the CUDA driver has to bounce pageable
//! transfers through its own internal pinned staging buffer, and the copy is
//! always fully synchronous (`cuMemcpyHtoD_v2`/`cuMemcpyDtoH_v2`). Allocating
//! the host side directly as pinned memory via `cuMemAllocHost_v2` removes
//! that extra bounce (full PCIe DMA bandwidth) and makes the async copy
//! variants (`cuMemcpyHtoDAsync_v2`/`cuMemcpyDtoHAsync_v2`) genuinely
//! asynchronous instead of silently degrading to synchronous behavior on
//! pageable memory.

use crate::infrastructure::device::{CudaContext, cuda_byte_count};
use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{HephaestusError, Result};
use std::sync::Arc;

/// A page-locked host buffer of `len` elements of `T`, freed via
/// `cuMemFreeHost` on drop.
///
/// Behaves like a `Vec<T>` for host-side reads/writes (`Deref`/`DerefMut` to
/// `[T]`) but the backing allocation is DMA-eligible, so passing its pointer
/// to `cuMemcpyHtoDAsync_v2`/`cuMemcpyDtoHAsync_v2` transfers at full
/// bandwidth instead of the driver's implicit pageable-memory staging copy.
pub(crate) struct PinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
    context: Arc<CudaContext>,
    marker: PhantomData<T>,
}

// SAFETY: `PinnedHostBuffer<T>` uniquely owns a page-locked host allocation
// (never aliased — no other handle to `ptr` exists) and carries an `Arc` for
// the drop-time context bind; sending it across threads is sound whenever
// `T` itself is `Send`, exactly like `Vec<T>`. Likewise `&PinnedHostBuffer<T>`
// is safe to share across threads whenever `T: Sync`, matching `Vec<T>`.
unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedHostBuffer<T> {}

impl<T: Pod> PinnedHostBuffer<T> {
    /// Allocate a zero-initialized pinned host buffer of `len` elements.
    ///
    /// # Errors
    /// Returns [`HephaestusError::AllocationFailed`] if the byte size
    /// overflows `usize` or the driver allocation fails (e.g. the host has
    /// exhausted its page-lockable memory budget).
    pub(crate) fn zeroed(context: Arc<CudaContext>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                context,
                marker: PhantomData,
            });
        }

        context.bind()?;
        let byte_len = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            HephaestusError::AllocationFailed {
                message: format!("pinned host buffer of {len} elements overflows byte size"),
            }
        })?;
        let byte_count = cuda_byte_count(byte_len, "pinned host buffer byte size")?;

        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: the context is current (`bind` above); `raw` is a valid
        // out-pointer for one pointer-sized value; `byte_count` is the exact
        // byte size requested, matching `cuMemAllocHost_v2`'s contract.
        let res = unsafe { cuda_oxide::sys::cuMemAllocHost_v2(&mut raw, byte_count) };
        if res != 0 || raw.is_null() {
            return Err(HephaestusError::AllocationFailed {
                message: format!("cuMemAllocHost_v2({byte_len} bytes) failed with code {res}"),
            });
        }

        // SAFETY: `raw` points to `byte_len` freshly allocated bytes with no
        // prior initialization; writing zero bytes over them is always sound,
        // and `T: Pod` guarantees the all-zero bit pattern is a valid `T`, so
        // reinterpreting the allocation as `[T; len]` below is well-typed.
        unsafe {
            std::ptr::write_bytes(raw.cast::<u8>(), 0, byte_len);
        }

        Ok(Self {
            ptr: raw.cast::<T>(),
            len,
            context,
            marker: PhantomData,
        })
    }

    /// Raw pointer for a `cuMemcpy*` call that writes into this buffer (e.g.
    /// as the destination of a device-to-host transfer). Takes `&mut self`
    /// so the write-through-raw-pointer is backed by a genuine unique
    /// borrow, not one derived from a shared reference.
    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.ptr.cast::<std::ffi::c_void>()
    }
}

impl<T: Pod> core::ops::Deref for PinnedHostBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is non-null (the `len == 0` case returns above),
        // points to `len` initialized `T` values (zeroed at construction,
        // subsequently only ever overwritten with valid `T` bytes via
        // `DerefMut`/`cuMemcpy*` into this same allocation), and this shared
        // borrow's lifetime is tied to `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Pod> core::ops::DerefMut for PinnedHostBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as `deref` above, with unique access via `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if self.context.bind().is_ok() {
            // SAFETY: `self.ptr` is non-null and was returned by
            // `cuMemAllocHost_v2` in this context; this buffer owns that
            // allocation exactly once and never frees it elsewhere.
            let res =
                unsafe { cuda_oxide::sys::cuMemFreeHost(self.ptr.cast::<std::ffi::c_void>()) };
            debug_assert_eq!(res, 0, "cuMemFreeHost failed with code {res}");
        } else {
            debug_assert!(false, "PinnedHostBuffer drop: context bind failed");
        }
    }
}
