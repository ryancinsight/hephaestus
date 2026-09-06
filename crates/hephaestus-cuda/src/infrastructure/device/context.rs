use super::properties::device_attribute;
use crate::infrastructure::driver::{Attribute, Driver};
use core::ffi::c_void;
use hephaestus_core::{HephaestusError, Result};

/// Owned CUDA context with a process-lifetime driver table.
///
/// The raw context is intentionally crate-private: the CUDA driver owns device
/// substrate state, while application/kernel modules receive only raw
/// `CUdeviceptr` values through `CudaBuffer::raw`.
#[derive(Debug)]
pub(crate) struct CudaContext {
    raw: *mut c_void,
    pub(crate) driver: &'static Driver,
    /// Whether this device supports the stream-ordered allocator
    /// (`cuMemAllocAsync`/`cuMemFreeAsync`). Detected once at context creation
    /// and read by both the allocation and the free path, so a buffer is always
    /// released by the allocator that produced it.
    pub(crate) stream_ordered: bool,
}

impl CudaContext {
    pub(super) fn create(driver: &'static Driver, device: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `device` was returned by `cuDeviceGet`; `raw` is a valid
        // out-pointer for one CUDA context handle.
        let status = unsafe {
            (driver.context.create)(
                &mut raw, 4, // CU_CTX_SCHED_BLOCKING_SYNC in cuda.h.
                device,
            )
        };
        if status != 0 {
            return Err(HephaestusError::DeviceUnavailable {
                message: format!("cuCtxCreate_v2 for CUDA device {device} -> {status}"),
            });
        }

        let mut context = Self {
            raw,
            driver,
            stream_ordered: false,
        };
        // Query failure is a driver fault. The owned context destroys itself
        // on this error path; unsupported capability alone selects legacy allocation.
        context.stream_ordered = device_attribute(
            driver,
            device,
            Attribute::MemoryPoolsSupported,
            "memory_pools_supported",
        )? != 0;
        Ok(context)
    }

    pub(crate) fn bind(&self) -> Result<()> {
        // SAFETY: `self.raw` is a live CUDA context owned by this value; CUDA
        // contexts are current per host thread, and setting it current does
        // not transfer ownership.
        let status = unsafe { (self.driver.context.bind)(self.raw) };
        if status != 0 {
            tracing::error!(
                operation = "cuCtxSetCurrent",
                status,
                "CUDA context bind failed"
            );
            return Err(HephaestusError::TransferFailed {
                message: format!("cuCtxSetCurrent -> {status}"),
            });
        }
        Ok(())
    }
}

// SAFETY: `CUcontext` is an opaque driver handle. The CUDA driver permits a
// context to be made current on any host thread; all use sites bind before
// issuing driver calls and never dereference the handle in Rust.
unsafe impl Send for CudaContext {}
// SAFETY: shared references only copy the opaque handle into driver calls;
// the driver owns synchronization for context operations.
unsafe impl Sync for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        // SAFETY: this value uniquely owns the context; every resource retains
        // it through Arc, so no owned resource remains at this final drop.
        // CUDA 13.3 cuda.h:6486–6504 permits destruction without making the
        // context current. It pops only if this context is already current;
        // binding here would overwrite an unrelated current context.
        let status = unsafe { (self.driver.context.destroy)(self.raw) };
        if status != 0 {
            tracing::error!(
                operation = "cuCtxDestroy_v2",
                status,
                "CUDA context release failed; driver retains context"
            );
        }
    }
}

/// Snapshot of the context current on this thread, for restoring around a
/// drop-time bind.
///
/// RAII drops (`CudaBuffer`, `PinnedHostBuffer`, `SafeCachedKernel`) must
/// bind their owned allocation's context to free it. Leaving that context
/// current would silently retarget the dropping thread: a drop running
/// between a consumer's `device.bind()` and a raw-handle driver call (e.g.
/// `cuLaunchKernel` over `CudaBuffer::raw`) would redirect that call to the
/// dropped allocation's context. Capture before the bind, restore after,
/// so drops stay invisible to the thread's context state.
pub(crate) struct CurrentContext(*mut c_void, &'static Driver);

impl CurrentContext {
    /// Capture the context current on this thread (null when none is bound).
    ///
    /// A failed query reports its native status; resource drops then skip
    /// binding because no trustworthy prior context exists to restore.
    pub(crate) fn capture(driver: &'static Driver) -> Result<Self> {
        let mut raw: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer for one context handle; the
        // call only reads thread-local driver state.
        let status = unsafe { (driver.context.current)(&mut raw) };
        if status != 0 {
            tracing::error!(
                operation = "cuCtxGetCurrent",
                status,
                "CUDA context capture failed; resource retained"
            );
            return Err(HephaestusError::TransferFailed {
                message: format!("cuCtxGetCurrent -> {status}"),
            });
        }
        Ok(Self(raw, driver))
    }

    /// Restore the captured context as this thread's current context.
    pub(crate) fn restore(self) {
        // SAFETY: `self.0` is either null (legally clears the current
        // context) or an opaque handle that was current on this same thread
        // at capture time; the handle is never dereferenced in Rust, and the
        // driver validates it — a context destroyed since capture yields an
        // error status, not undefined behavior. Setting a context current
        // does not transfer ownership.
        let status = unsafe { (self.1.context.bind)(self.0) };
        if status != 0 {
            tracing::error!(
                operation = "cuCtxSetCurrent",
                status,
                "CUDA context restoration failed; aborting to preserve routing"
            );
            // Restoration failure makes thread-current routing untrustworthy.
            // Abort preserves routing invariants without panicking in Drop.
            std::process::abort();
        }
    }
}

#[cfg(test)]
mod tests;
