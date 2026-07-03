use core::ffi::c_void;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::Arc;

use mnemosyne::{
    CudaDeviceBackend, CudaHostPinnedBackend, CudaUnifiedBackend, MnemosyneAllocator,
    StandardPolicy,
};

pub(crate) static CUDA_DEVICE_ALLOCATOR: MnemosyneAllocator<StandardPolicy, CudaDeviceBackend> =
    MnemosyneAllocator::new();
pub(crate) static CUDA_HOST_PINNED_ALLOCATOR: MnemosyneAllocator<
    StandardPolicy,
    CudaHostPinnedBackend,
> = MnemosyneAllocator::new();
pub(crate) static CUDA_UNIFIED_ALLOCATOR: MnemosyneAllocator<StandardPolicy, CudaUnifiedBackend> =
    MnemosyneAllocator::new();

use bytemuck::Pod;
use hephaestus_core::{
    validate_buffer_size, validate_slice_alignment, ComputeDevice, ComputeDeviceAcquisition,
    ComputeDeviceCapabilities, DeviceFeature, DeviceLimits, DevicePreference, HephaestusError,
    Result,
};

use crate::infrastructure::buffer::{CudaBuffer, DevicePtr};

/// An acquired CUDA device.
///
/// Holds the cutile-rs (`cuda-core`) device handle for the default ordinal.
/// `Clone` is cheap (an `Arc` clone). Device acquisition mirrors coeus-cuda's
/// driver: the CUDA driver is dynamically loaded, so constructing this never
/// requires a CUDA toolkit at build time, only `nvcuda`/`libcuda` at runtime.
#[derive(Clone)]
pub struct CudaDevice {
    device: Arc<cuda_core::Device>,
    limits: DeviceLimits,
    features: CudaDeviceFeatures,
    /// Compiled-kernel cache. Slots hold only successful compilations;
    /// failures leave the `OnceLock` empty so the key can retry (see
    /// [`crate::application::pipeline::cached_kernel`]).
    pub(crate) pipeline_cache: Arc<
        moirai_sync::sync::ConcurrentHashMap<
            String,
            Arc<std::sync::OnceLock<Arc<crate::infrastructure::compiler::SafeCachedKernel>>>,
        >,
    >,
    topology: Option<Arc<themis::GpuTopology>>,
}

#[derive(Clone, Copy, Debug)]
struct CudaDeviceFeatures {
    shader_f64: bool,
    mappable_primary_buffers: bool,
    push_constants: bool,
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
        let device_ordinal = Self::default_device_ordinal()?;
        Self::try_with_ordinal(device_ordinal)
    }

    /// Acquire a CUDA device by ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::AdapterUnavailable`] when the CUDA driver or
    /// requested ordinal is unavailable. Returns
    /// [`HephaestusError::DeviceUnavailable`] when the context cannot be bound.
    pub fn try_with_ordinal(device_ordinal: usize) -> Result<Self> {
        if !mnemosyne_backend::is_cuda_available() {
            return Err(HephaestusError::AdapterUnavailable {
                message: "CUDA unified memory driver not available or initialization failed"
                    .to_string(),
            });
        }

        let device_ordinal = i32::try_from(device_ordinal).map_err(|_| {
            HephaestusError::AdapterUnavailable {
                message: format!("CUDA device ordinal {device_ordinal} exceeds i32 range"),
            }
        })?;

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
        let limits = query_device_limits(&device)?;
        let features = query_device_features(&device)?;
        let topology = Some(Arc::new(query_topology(&device)?));
        let dev = Self {
            device,
            limits,
            features,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            topology,
        };
        // Sanity check to confirm that dynamic memory mapping and copying is functional (fails closed on headless/stub drivers)
        {
            let buf =
                dev.alloc_zeroed::<u32>(1)
                    .map_err(|e| HephaestusError::AdapterUnavailable {
                        message: format!("CUDA sanity check allocation failed: {e:?}"),
                    })?;
            let host_val = [42u32];
            dev.write_buffer(&buf, &host_val)
                .map_err(|e| HephaestusError::AdapterUnavailable {
                    message: format!("CUDA sanity check write failed: {e:?}"),
                })?;
            let mut read_val = [0u32];
            dev.download(&buf, &mut read_val)
                .map_err(|e| HephaestusError::AdapterUnavailable {
                    message: format!("CUDA sanity check download failed: {e:?}"),
                })?;
            if read_val[0] != 42 {
                return Err(HephaestusError::AdapterUnavailable {
                    message: "CUDA sanity check data mismatch".to_string(),
                });
            }
        }
        Ok(dev)
    }

    fn default_device_ordinal() -> Result<usize> {
        let mut device_ordinal = 0usize;
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
                // SAFETY: `count` is a valid out-pointer for one `c_int`; the
                // CUDA driver has been initialized by `is_cuda_available`.
                if unsafe { cuda_core::sys::cuDeviceGetCount(&mut count) } == 0 && count > 0 {
                    device_ordinal = usize::try_from(thread_id % count).unwrap_or(0);
                }
            }
        }

        Ok(device_ordinal)
    }

    fn device_count() -> Result<usize> {
        if !mnemosyne_backend::is_cuda_available() {
            return Err(HephaestusError::AdapterUnavailable {
                message: "CUDA unified memory driver not available or initialization failed"
                    .to_string(),
            });
        }
        let mut count: core::ffi::c_int = 0;
        // SAFETY: `count` is a valid out-pointer for one `c_int`; the CUDA
        // driver has been initialized by `is_cuda_available`.
        let status = unsafe { cuda_core::sys::cuDeviceGetCount(&mut count) };
        if status != 0 {
            return Err(HephaestusError::AdapterUnavailable {
                message: format!("CUDA device count query failed with status {status}"),
            });
        }
        usize::try_from(count).map_err(|_| HephaestusError::AdapterUnavailable {
            message: format!("CUDA device count {count} is negative"),
        })
    }

    fn require_limits(actual: DeviceLimits, required: DeviceLimits) -> Result<()> {
        let comparable = [
            (
                "max_buffer_size",
                actual.max_buffer_size,
                required.max_buffer_size,
            ),
            (
                "max_compute_workgroup_size_x",
                u64::from(actual.max_compute_workgroup_size_x),
                u64::from(required.max_compute_workgroup_size_x),
            ),
            (
                "max_compute_workgroup_size_y",
                u64::from(actual.max_compute_workgroup_size_y),
                u64::from(required.max_compute_workgroup_size_y),
            ),
            (
                "max_compute_workgroup_size_z",
                u64::from(actual.max_compute_workgroup_size_z),
                u64::from(required.max_compute_workgroup_size_z),
            ),
            (
                "max_compute_invocations_per_workgroup",
                u64::from(actual.max_compute_invocations_per_workgroup),
                u64::from(required.max_compute_invocations_per_workgroup),
            ),
            (
                "max_compute_workgroup_storage_size",
                u64::from(actual.max_compute_workgroup_storage_size),
                u64::from(required.max_compute_workgroup_storage_size),
            ),
            (
                "max_push_constant_size",
                u64::from(actual.max_push_constant_size),
                u64::from(required.max_push_constant_size),
            ),
        ];
        for (name, available, needed) in comparable {
            if available < needed {
                return Err(HephaestusError::DeviceUnavailable {
                    message: format!(
                        "CUDA device limit {name} {available} is below required {needed}"
                    ),
                });
            }
        }
        if let (Some(available), Some(needed)) = (
            actual.max_storage_buffers_per_shader_stage,
            required.max_storage_buffers_per_shader_stage,
        ) {
            if available < needed {
                return Err(HephaestusError::DeviceUnavailable {
                    message: format!(
                        "CUDA shader-stage storage-buffer limit {available} is below required {needed}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// The device topology snapshot captured at acquisition, when available.
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.topology.as_deref()
    }

    /// The underlying cutile-rs device handle (module lifetime management).
    #[inline]
    pub(crate) fn cu_device(&self) -> &Arc<cuda_core::Device> {
        &self.device
    }

    /// Bind the device context to the current thread before a driver call.
    ///
    /// Transfers, allocations, module loads, and kernel launches execute
    /// against the thread's current context; binding makes this device's
    /// context current (CUDA contexts are thread-affine), so calls from any
    /// thread target the right device.
    pub fn bind(&self) -> Result<()> {
        self.device
            .bind_to_thread()
            .map_err(|e| HephaestusError::TransferFailed {
                message: format!("bind device to thread: {e:?}"),
            })
    }

    /// Allocate `bytes` of device memory according to the tier.
    fn alloc_bytes_with_tier(&self, bytes: usize, tier: themis::MemoryTier) -> Result<DevicePtr> {
        self.bind()?;

        let layout =
            Layout::from_size_align(bytes, 4).map_err(|e| HephaestusError::AllocationFailed {
                message: format!("invalid allocation layout (size {bytes}): {e:?}"),
            })?;

        let ptr_val = match tier {
            themis::MemoryTier::HostPinned => unsafe { CUDA_HOST_PINNED_ALLOCATOR.alloc(layout) },
            themis::MemoryTier::Dram => unsafe { CUDA_UNIFIED_ALLOCATOR.alloc(layout) },
            _ => unsafe { CUDA_DEVICE_ALLOCATOR.alloc(layout) },
        };

        if ptr_val.is_null() {
            return Err(HephaestusError::AllocationFailed {
                message: format!("Mnemosyne failed to allocate {bytes} bytes for tier {tier:?}"),
            });
        }

        Ok(ptr_val as DevicePtr)
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

fn device_attribute(
    device: &cuda_core::Device,
    attribute: cuda_core::sys::CUdevice_attribute,
    name: &str,
) -> Result<i32> {
    let mut value: core::ffi::c_int = 0;
    // SAFETY: `device.cu_device()` is a valid `CUdevice` handle returned by
    // cuda_core for an acquired CUDA device; `value` is a valid out-pointer for
    // one `c_int`.
    let result =
        unsafe { cuda_core::sys::cuDeviceGetAttribute(&mut value, attribute, device.cu_device()) };
    if result != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuDeviceGetAttribute({name}) -> {result}"),
        });
    }
    Ok(value)
}

fn current_memory_info() -> Result<(usize, usize)> {
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    // SAFETY: the CUDA context is current for the calling thread at each call
    // site; both pointers address one writable `usize`.
    let result = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes) };
    if result != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuMemGetInfo_v2 -> {result}"),
        });
    }
    Ok((free_bytes, total_bytes))
}

fn nonnegative_u32(value: i32) -> u32 {
    u32::try_from(value).unwrap_or(0)
}

fn query_device_limits(device: &cuda_core::Device) -> Result<DeviceLimits> {
    use cuda_core::sys;

    let (free_bytes, _) = current_memory_info()?;
    Ok(DeviceLimits {
        max_buffer_size: free_bytes as u64,
        max_compute_workgroup_size_x: nonnegative_u32(device_attribute(
            device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            "max_block_dim_x",
        )?),
        max_compute_workgroup_size_y: nonnegative_u32(device_attribute(
            device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            "max_block_dim_y",
        )?),
        max_compute_workgroup_size_z: nonnegative_u32(device_attribute(
            device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            "max_block_dim_z",
        )?),
        max_compute_invocations_per_workgroup: nonnegative_u32(device_attribute(
            device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            "max_threads_per_block",
        )?),
        max_compute_workgroup_storage_size: nonnegative_u32(device_attribute(
            device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            "max_shared_memory_per_block",
        )?),
        max_storage_buffers_per_shader_stage: None,
        max_push_constant_size: 0,
    })
}

fn query_device_features(device: &cuda_core::Device) -> Result<CudaDeviceFeatures> {
    use cuda_core::sys;

    let major = device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        "compute_capability_major",
    )?;
    let minor = device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        "compute_capability_minor",
    )?;
    let mappable = device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        "can_map_host_memory",
    )? != 0;
    let unified_addressing = device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        "unified_addressing",
    )? != 0;
    let compute_capability = major * 10 + minor;

    Ok(CudaDeviceFeatures {
        shader_f64: compute_capability >= 13,
        mappable_primary_buffers: mappable || unified_addressing,
        push_constants: true,
    })
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

    let compute_units = nonnegative_u32(device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        "multiprocessor_count",
    )?);
    let warp_width = nonnegative_u32(device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        "warp_size",
    )?);
    let max_threads_per_unit = nonnegative_u32(device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        "max_threads_per_multiprocessor",
    )?);
    let registers_per_unit = nonnegative_u32(device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        "max_registers_per_multiprocessor",
    )?);
    let shared_mem_per_unit_bytes = nonnegative_u32(device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        "max_shared_memory_per_multiprocessor",
    )?) as usize;
    let l2_bytes = nonnegative_u32(device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        "l2_cache_size",
    )?) as usize;

    // Total device memory via `cuMemGetInfo_v2` (memory-management family,
    // current-context query) rather than `cuDeviceTotalMem_v2`: the latter's
    // dynamic symbol is unresolved in this cutile-rs binding and faults when
    // called, whereas the memory-family entry points (e.g. `cuMemAlloc_v2`)
    // resolve correctly. The context was made current by `bind_to_thread`.
    let (_, total_bytes) = current_memory_info()?;

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

impl ComputeDeviceCapabilities for CudaDevice {
    #[inline]
    fn device_limits(&self) -> DeviceLimits {
        self.limits
    }

    #[inline]
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        match feature {
            DeviceFeature::TimestampQuery => false,
            DeviceFeature::ShaderF64 => self.features.shader_f64,
            DeviceFeature::ShaderF16 => false,
            DeviceFeature::MappablePrimaryBuffers => self.features.mappable_primary_buffers,
            DeviceFeature::PushConstants => self.features.push_constants,
        }
    }
}

impl ComputeDeviceAcquisition for CudaDevice {
    fn try_acquire_device(
        _label: &str,
        _device_preference: DevicePreference,
        _optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self> {
        let device = Self::try_default()?;
        Self::require_limits(device.device_limits(), required_limits)?;
        Ok(device)
    }

    fn try_acquire_devices(
        _label_prefix: &str,
        max_devices: usize,
        _device_preference: DevicePreference,
        _optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Vec<Self>> {
        let count = Self::device_count()?;
        let mut devices = Vec::with_capacity(count.min(max_devices));
        for ordinal in 0..count.min(max_devices) {
            let device = Self::try_with_ordinal(ordinal)?;
            Self::require_limits(device.device_limits(), required_limits)?;
            devices.push(device);
        }
        Ok(devices)
    }
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

    #[inline]
    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        CudaDevice::write_sub_buffer(self, buffer, offset, host)
    }

    fn synchronize(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: the CUDA context is current for this thread after `bind`.
        let res = unsafe { cuda_core::sys::cuCtxSynchronize() };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("cuCtxSynchronize -> {res}"),
            });
        }
        Ok(())
    }
}
