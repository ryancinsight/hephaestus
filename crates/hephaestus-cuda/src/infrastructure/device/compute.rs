use super::CudaDevice;
use crate::infrastructure::buffer::CudaBuffer;
use core::ffi::c_void;
use eunomia::Pod;
use hephaestus_core::{
    CommandStream, ComputeDevice, HephaestusError, KernelDevice, Result, validate_buffer_size,
    validate_slice_alignment,
};

impl ComputeDevice for CudaDevice {
    type Buffer<T: Pod> = CudaBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "cuda"
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let buffer = self.alloc_uninitialized_with_hint::<T>(len, hint)?;
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            HephaestusError::AllocationFailed {
                message: format!("byte count overflow for {len} elements"),
            }
        })?;
        if bytes == 0 {
            return Ok(buffer);
        }

        // SAFETY: `buffer.raw()` addresses `bytes` of device memory owned by
        // this buffer, and the checked byte count covers the allocation.
        let res = unsafe { (self.driver().memory.fill)(buffer.raw(), 0, bytes) };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("zero-init cuMemsetD8_v2 -> {res}"),
            });
        }
        Ok(buffer)
    }

    fn alloc_uninitialized_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        validate_buffer_size::<T>(len)?;
        let tier = Self::allocation_tier(hint)?;
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            HephaestusError::AllocationFailed {
                message: format!("byte count overflow for {len} elements"),
            }
        })?;
        if bytes == 0 {
            return Ok(CudaBuffer::new(0, len, tier, self.context.clone()));
        }
        let ptr = self.alloc_bytes(bytes)?;
        Ok(CudaBuffer::new(ptr, len, tier, self.context.clone()))
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        validate_slice_alignment(host)?;
        let len = host.len();
        let tier = Self::allocation_tier(hint)?;
        let bytes = core::mem::size_of_val(host);
        if bytes == 0 {
            return Ok(CudaBuffer::new(0, len, tier, self.context.clone()));
        }
        let ptr = self.alloc_bytes(bytes)?;

        // SAFETY: `ptr` addresses `bytes` of device memory just allocated;
        // `host` is `bytes` of readable host memory (`T: Pod`). The buffer owns
        // `ptr`, so it is freed if the copy fails.
        let res = unsafe {
            (self.driver().memory.upload)(ptr, host.as_ptr().cast::<core::ffi::c_void>(), bytes)
        };
        let buffer = CudaBuffer::<T>::new(ptr, len, tier, self.context.clone());
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("upload cuMemcpyHtoD_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(buffer)
    }

    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()> {
        validate_slice_alignment(out)?;
        if out.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: out.len(),
                device_len: buffer.len,
            });
        }
        let bytes = core::mem::size_of_val(out);
        if bytes == 0 {
            return Ok(());
        }

        self.bind()?;
        // SAFETY: `buffer.ptr` addresses `bytes` of device memory (len matches,
        // checked above); `out` is `bytes` of writable host memory (`T: Pod`).
        let res = unsafe {
            (self.driver().memory.download)(out.as_mut_ptr().cast::<c_void>(), buffer.ptr, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("download cuMemcpyDtoH_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }

    fn download_owned<T: Pod>(&self, buffer: &Self::Buffer<T>) -> Result<Vec<T>> {
        let len = buffer.len;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|error| HephaestusError::AllocationFailed {
                message: format!(
                    "CUDA host download allocation for {len} elements failed: {error}"
                ),
            })?;
        if core::mem::size_of::<T>() == 0 {
            out.resize(len, eunomia::Zeroable::zeroed());
            return Ok(out);
        }
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            HephaestusError::AllocationFailed {
                message: format!("CUDA host download byte count overflows for {len} elements"),
            }
        })?;
        if bytes != 0 {
            self.bind()?;
            // SAFETY: `try_reserve_exact` established capacity for `len`
            // elements, so the spare-capacity pointer addresses `bytes`
            // writable host bytes. The synchronous copy initializes every byte
            // before vector length is published below.
            let status = unsafe {
                (self.driver().memory.download)(
                    out.spare_capacity_mut().as_mut_ptr().cast::<c_void>(),
                    buffer.ptr,
                    bytes,
                )
            };
            if status != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!("owned download cuMemcpyDtoH_v2({bytes} bytes) -> {status}"),
                });
            }
        }
        // SAFETY: `T` is non-zero-sized here and the successful synchronous
        // copy above initialized all `len` elements before publication.
        unsafe { out.set_len(len) };
        Ok(out)
    }

    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()> {
        validate_slice_alignment(host)?;
        if host.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: buffer.len,
            });
        }
        if host.is_empty() || core::mem::size_of::<T>() == 0 {
            return Ok(());
        }
        self.bind()?;
        let bytes = std::mem::size_of_val(host);

        // SAFETY: `buffer.ptr` is a valid device pointer allocated by this
        // device; `host` is `bytes` of readable host memory (`T: Pod`).
        let res = unsafe {
            (self.driver().memory.upload)(buffer.raw(), host.as_ptr() as *const c_void, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("write_buffer cuMemcpyHtoD_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }

    #[inline]
    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        CudaDevice::write_sub_buffer(self, buffer, offset, host)
    }

    fn copy_buffer<T: Pod>(&self, src: &CudaBuffer<T>, dst: &CudaBuffer<T>) -> Result<()> {
        let mut stream = self.stream()?;
        stream.copy(src, dst)?;
        stream.submit()?;
        self.synchronize_default_stream()
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        CudaDevice::topology(self)
    }

    fn synchronize(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: the CUDA context is current for this thread after `bind`.
        let res = unsafe { (self.driver().context.synchronize)() };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("cuCtxSynchronize -> {res}"),
            });
        }
        Ok(())
    }
}

impl CudaDevice {
    /// Wait for every operation enqueued on the legacy/null default stream.
    ///
    /// Crate-internal barrier for async-copy paths whose enqueued transfers
    /// must not outlive their enclosing frame (2-D region copies drain it on
    /// every exit, error exits included).
    pub(crate) fn synchronize_default_stream(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: this device's context is current after `bind`; a null stream
        // handle denotes the default stream used by synchronous-form copies.
        let res = unsafe { (self.driver().context.synchronize_stream)(core::ptr::null_mut()) };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("cuStreamSynchronize(default) -> {res}"),
            });
        }
        Ok(())
    }
}
