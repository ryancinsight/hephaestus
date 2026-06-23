use core::ffi::c_void;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    validate_buffer_size, validate_slice_alignment, ComputeDevice, HephaestusError, Result,
};

use crate::infrastructure::buffer::{CudaBuffer, DevicePtr};

/// An acquired CUDA device.
///
/// Holds the cutile-rs (`cuda-core`) device handle for the default ordinal.
/// `Clone` is cheap (an `Arc` clone). Device acquisition mirrors coeus-cuda's
/// driver: the CUDA driver is dynamically loaded, so constructing this never
/// requires a CUDA toolkit at build time, only `nvcuda`/`libcuda` at runtime.
#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct CudaDevice {
    device: Arc<cuda_core::Device>,
    pub(crate) pipeline_cache: Arc<
        moirai_sync::sync::ConcurrentHashMap<
            String,
            Arc<
                std::sync::OnceLock<
                    std::result::Result<
                        Arc<crate::infrastructure::compiler::SafeCachedKernel>,
                        String,
                    >,
                >,
            >,
        >,
    >,
    topology: Option<Arc<themis::GpuTopology>>,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaDevice").finish_non_exhaustive()
    }
}

impl CudaDevice {
    /// Acquire the default CUDA device (ordinal 0).
    ///
    /// Returns [`HephaestusError::AdapterUnavailable`] when no CUDA driver or
    /// device is present, rather than fabricating a device. The acquired
    /// device is bound to the calling thread.
    pub fn try_default() -> Result<Self> {
        if !mnemosyne_backend::is_cuda_available() {
            return Err(HephaestusError::AdapterUnavailable {
                message: "CUDA unified memory driver not available or initialization failed"
                    .to_string(),
            });
        }

        let mut device_ordinal = 0;
        if let Ok(thread_id_str) = std::env::var("NEXTEST_THREAD_ID") {
            if let Ok(thread_id) = thread_id_str.parse::<i32>() {
                static STAGGERED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !STAGGERED.load(std::sync::atomic::Ordering::Acquire)
                    && STAGGERED
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        )
                        .is_ok()
                {
                    std::thread::sleep(std::time::Duration::from_millis(thread_id as u64 * 100));
                }

                let mut count: core::ffi::c_int = 0;
                if unsafe { cuda_core::sys::cuDeviceGetCount(&mut count) } == 0 && count > 0 {
                    device_ordinal = thread_id % count;
                }
            }
        }

        let device = cuda_async::device_context::with_device(device_ordinal as usize, |device| {
            device.clone()
        })
        .map_err(|e| HephaestusError::AdapterUnavailable {
            message: format!("CUDA device {device_ordinal} unavailable: {e:?}"),
        })?;
        device
            .bind_to_thread()
            .map_err(|e| HephaestusError::DeviceUnavailable {
                message: format!("bind device {device_ordinal} to thread: {e:?}"),
            })?;
        let topology = Some(Arc::new(query_topology(&device)?));
        Ok(Self {
            device,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            topology,
        })
    }

    /// The device topology snapshot captured at acquisition, when available.
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.topology.as_deref()
    }

    /// Bind the device context to the current thread before a driver call.
    ///
    /// Transfers and allocations execute against the thread's current context;
    /// binding makes this device's context current (CUDA contexts are
    /// thread-affine), so calls from any thread target the right device.
    pub fn bind(&self) -> Result<()> {
        self.device
            .bind_to_thread()
            .map_err(|e| HephaestusError::TransferFailed {
                message: format!("bind device to thread: {e:?}"),
            })
    }

    /// Allocate `bytes` of device memory according to the tier.
    fn alloc_bytes_with_tier(&self, bytes: usize, tier: themis::MemoryTier) -> Result<DevicePtr> {
        let mut ptr: DevicePtr = 0;
        self.bind()?;

        match tier {
            themis::MemoryTier::HostPinned => {
                let mut host_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
                // SAFETY: `host_ptr` is a valid out-pointer for host allocations; the
                // context is current. Flag 0x02 is CU_MEMHOSTALLOC_DEVICEMAP.
                let res = unsafe {
                    cuda_core::sys::cuMemHostAlloc(core::ptr::addr_of_mut!(host_ptr), bytes, 0x02)
                };
                if res != 0 {
                    return Err(HephaestusError::AllocationFailed {
                        message: format!("cuMemHostAlloc({bytes} bytes) -> {res}"),
                    });
                }
                ptr = host_ptr as DevicePtr;
            }
            themis::MemoryTier::Dram => {
                // SAFETY: `ptr` is a valid out-pointer for a single `CUdeviceptr`;
                // the context is current. Flag 0x01 is CU_MEMATTACH_GLOBAL.
                let res = unsafe {
                    cuda_core::sys::cuMemAllocManaged(
                        core::ptr::addr_of_mut!(ptr).cast(),
                        bytes,
                        0x01,
                    )
                };
                if res != 0 {
                    return Err(HephaestusError::AllocationFailed {
                        message: format!("cuMemAllocManaged({bytes} bytes) -> {res}"),
                    });
                }
            }
            _ => {
                // SAFETY: `ptr` is a valid out-pointer for a single `CUdeviceptr`; the
                // context is current.
                let res = unsafe {
                    cuda_core::sys::cuMemAlloc_v2(core::ptr::addr_of_mut!(ptr).cast(), bytes)
                };
                if res != 0 {
                    return Err(HephaestusError::AllocationFailed {
                        message: format!("cuMemAlloc_v2({bytes} bytes) -> {res}"),
                    });
                }
            }
        }
        Ok(ptr)
    }

    /// Copy a subset of a device buffer's contents into a host slice (device→host).
    pub fn download_sub_buffer<T: Pod>(
        &self,
        buffer: &CudaBuffer<T>,
        offset: usize,
        out: &mut [T],
    ) -> Result<()> {
        validate_slice_alignment(out)?;
        let end =
            offset
                .checked_add(out.len())
                .ok_or_else(|| HephaestusError::AllocationFailed {
                    message: format!("offset {offset} + out.len() {} overflows usize", out.len()),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        if out.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let element_size = std::mem::size_of::<T>();
        let byte_offset = (offset as u64)
            .checked_mul(element_size as u64)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!("byte offset calculation overflows u64 for offset {offset}"),
            })?;
        let bytes = std::mem::size_of_val(out);
        let src_ptr = buffer.raw() + byte_offset;
        // SAFETY: `src_ptr` is a valid device pointer offset from a pointer allocated by this device;
        // `out` is `bytes` of writable host memory (`T: Pod`).
        let res = unsafe {
            cuda_core::sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, src_ptr, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("download_sub_buffer cuMemcpyDtoH_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }

    /// Overwrite a subset of a device buffer with host data (host→device).
    pub fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &CudaBuffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        validate_slice_alignment(host)?;
        let end =
            offset
                .checked_add(host.len())
                .ok_or_else(|| HephaestusError::AllocationFailed {
                    message: format!(
                        "offset {offset} + host.len() {} overflows usize",
                        host.len()
                    ),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        if host.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let element_size = std::mem::size_of::<T>();
        let byte_offset = (offset as u64)
            .checked_mul(element_size as u64)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!("byte offset calculation overflows u64 for offset {offset}"),
            })?;
        let bytes = std::mem::size_of_val(host);
        let dest_ptr = buffer.raw() + byte_offset;
        // SAFETY: `dest_ptr` is a valid device pointer offset from a pointer allocated by this device;
        // `host` is `bytes` of readable host memory (`T: Pod`).
        let res = unsafe {
            cuda_core::sys::cuMemcpyHtoD_v2(dest_ptr, host.as_ptr() as *const c_void, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("write_sub_buffer cuMemcpyHtoD_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }
}

/// Query real device properties for the themis topology snapshot.
///
/// hephaestus is the stack's provider of GPU device properties into themis
/// `GpuTopology` (atlas ADR 0002); the placement law consumes these, so the
/// values are read from the driver via `cuDeviceGetAttribute` /
/// `cuDeviceTotalMem` rather than assumed. A failed attribute read on a device
/// that was just acquired and bound indicates a broken device, surfaced as
/// [`HephaestusError::DeviceUnavailable`].
fn query_topology(device: &cuda_core::Device) -> Result<themis::GpuTopology> {
    use cuda_core::sys;

    let cu = device.cu_device();
    let attr = |a: sys::CUdevice_attribute, what: &str| -> Result<i32> {
        let mut value: core::ffi::c_int = 0;
        // SAFETY: `cu` is a valid `CUdevice` handle returned by cuda_core for a
        // device acquired and bound above; `value` is a valid out-pointer for
        // one `c_int`.
        let res = unsafe { sys::cuDeviceGetAttribute(&mut value, a, cu) };
        if res != 0 {
            return Err(HephaestusError::DeviceUnavailable {
                message: format!("cuDeviceGetAttribute({what}) -> {res}"),
            });
        }
        Ok(value)
    };
    let nonneg = |v: i32| u32::try_from(v).unwrap_or(0);

    let compute_units = nonneg(attr(
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        "multiprocessor_count",
    )?);
    let warp_width = nonneg(attr(
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        "warp_size",
    )?);
    let max_threads_per_unit = nonneg(attr(
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        "max_threads_per_multiprocessor",
    )?);
    let registers_per_unit = nonneg(attr(
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        "max_registers_per_multiprocessor",
    )?);
    let shared_mem_per_unit_bytes = nonneg(attr(
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        "max_shared_memory_per_multiprocessor",
    )?) as usize;
    let l2_bytes = nonneg(attr(
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        "l2_cache_size",
    )?) as usize;

    // Total device memory via `cuMemGetInfo_v2` (memory-management family,
    // current-context query) rather than `cuDeviceTotalMem_v2`: the latter's
    // dynamic symbol is unresolved in this cutile-rs binding and faults when
    // called, whereas the memory-family entry points (e.g. `cuMemAlloc_v2`)
    // resolve correctly. The context was made current by `bind_to_thread`.
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    // SAFETY: the device context is current (bound above); `free_bytes` and
    // `total_bytes` are valid out-pointers for one `usize` each.
    let res = unsafe { sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes) };
    if res != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuMemGetInfo_v2 -> {res}"),
        });
    }

    Ok(themis::GpuTopology::from_provider(
        themis::GpuDeviceProperties {
            compute_units,
            warp_width,
            max_threads_per_unit,
            registers_per_unit,
            shared_mem_per_unit_bytes,
            l2_bytes,
            memory_tier: themis::MemoryTier::Device,
            memory_bytes: total_bytes as u64,
        },
    ))
}

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
        validate_buffer_size::<T>(len)?;
        let tier = match hint {
            themis::PlacementHint::Tier(t) => t,
            _ => themis::MemoryTier::Device,
        };
        if len == 0 {
            return Ok(CudaBuffer::new(0, 0, tier));
        }
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            HephaestusError::AllocationFailed {
                message: format!("byte count overflow for {len} elements"),
            }
        })?;
        let ptr = self.alloc_bytes_with_tier(bytes, tier)?;
        // SAFETY: `ptr` addresses `bytes` of device memory just allocated.
        let res = unsafe { cuda_core::sys::cuMemsetD8_v2(ptr, 0, bytes) };
        let buffer = CudaBuffer::<T>::new(ptr, len, tier);
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("zero-init cuMemsetD8_v2 -> {res}"),
            });
        }
        Ok(buffer)
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        validate_slice_alignment(host)?;
        let len = host.len();
        let tier = match hint {
            themis::PlacementHint::Tier(t) => t,
            _ => themis::MemoryTier::Device,
        };
        if len == 0 {
            return Ok(CudaBuffer::new(0, 0, tier));
        }
        let bytes = core::mem::size_of_val(host);
        let ptr = self.alloc_bytes_with_tier(bytes, tier)?;
        // SAFETY: `ptr` addresses `bytes` of device memory just allocated;
        // `host` is `bytes` of readable host memory (`T: Pod`). The buffer owns
        // `ptr`, so it is freed if the copy fails.
        let res = unsafe {
            cuda_core::sys::cuMemcpyHtoD_v2(ptr, host.as_ptr().cast::<core::ffi::c_void>(), bytes)
        };
        let buffer = CudaBuffer::<T>::new(ptr, len, tier);
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
        if buffer.len == 0 {
            return Ok(());
        }
        let bytes = core::mem::size_of_val(out);
        self.bind()?;
        // SAFETY: `buffer.ptr` addresses `bytes` of device memory (len matches,
        // checked above); `out` is `bytes` of writable host memory (`T: Pod`).
        let res = unsafe {
            cuda_core::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast::<c_void>(), buffer.ptr, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("download cuMemcpyDtoH_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }

    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()> {
        validate_slice_alignment(host)?;
        if host.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: buffer.len,
            });
        }
        if host.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let bytes = std::mem::size_of_val(host);
        // SAFETY: `buffer.ptr` is a valid device pointer allocated by this
        // device; `host` is `bytes` of readable host memory (`T: Pod`).
        let res = unsafe {
            cuda_core::sys::cuMemcpyHtoD_v2(buffer.raw(), host.as_ptr() as *const c_void, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("write_buffer cuMemcpyHtoD_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }
}
