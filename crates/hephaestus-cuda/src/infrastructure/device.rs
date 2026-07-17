use core::ffi::c_void;
use std::sync::Arc;

use bytemuck::Pod;
use cuda_oxide::Cuda;
use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceFeature,
    DeviceLimits, DevicePreference, HephaestusError, Result, validate_buffer_size,
    validate_slice_alignment,
};

use crate::application::pipeline::PipelineKey;
use crate::infrastructure::buffer::{CudaBuffer, DevicePtr};

/// CUDA context handle acquired through cuda-oxide's driver bindings.
///
/// The raw context is intentionally crate-private: cuda-oxide owns device
/// substrate state, while application/kernel modules receive only raw
/// `CUdeviceptr` values through `CudaBuffer::raw`.
#[derive(Debug)]
pub(crate) struct CudaContext {
    raw: cuda_oxide::sys::CUcontext,
}

impl CudaContext {
    fn create(device: cuda_oxide::sys::CUdevice) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `device` was returned by `cuDeviceGet`; `raw` is a valid
        // out-pointer for one CUDA context handle.
        let status = unsafe {
            cuda_oxide::sys::cuCtxCreate_v2(
                &mut raw,
                cuda_oxide::sys::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC,
                device,
            )
        };
        if status != 0 {
            return Err(HephaestusError::DeviceUnavailable {
                message: format!("cuCtxCreate_v2 for CUDA device {device} -> {status}"),
            });
        }
        Ok(Self { raw })
    }

    pub(crate) fn bind(&self) -> Result<()> {
        // SAFETY: `self.raw` is a live CUDA context owned by this value; CUDA
        // contexts are current per host thread, and setting it current does
        // not transfer ownership.
        let status = unsafe { cuda_oxide::sys::cuCtxSetCurrent(self.raw) };
        if status != 0 {
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
        if self.bind().is_ok() {
            // SAFETY: `self.raw` is the live context owned by this value and
            // is current on this thread after `bind`.
            let status = unsafe { cuda_oxide::sys::cuCtxDestroy_v2(self.raw) };
            debug_assert_eq!(status, 0, "cuCtxDestroy_v2 failed with code {status}");
        } else {
            debug_assert!(false, "CudaContext drop: context bind failed");
        }
    }
}

/// An acquired CUDA device.
///
/// Holds a cuda-oxide-created context for the selected device ordinal.
/// `Clone` is cheap (an `Arc` clone). Device acquisition mirrors coeus-cuda's
/// driver: the CUDA driver is dynamically loaded, so constructing this never
/// requires a CUDA toolkit at build time, only `nvcuda`/`libcuda` at runtime.
#[derive(Clone)]
pub struct CudaDevice {
    context: Arc<CudaContext>,
    limits: DeviceLimits,
    features: CudaDeviceFeatures,
    /// Compiled-kernel cache. Slots hold only successful compilations;
    /// failures leave the `OnceLock` empty so the key can retry (see
    /// [`crate::application::pipeline::cached_kernel`]).
    pub(crate) pipeline_cache: Arc<
        moirai_sync::sync::ConcurrentHashMap<
            PipelineKey,
            Arc<std::sync::OnceLock<Arc<crate::infrastructure::compiler::SafeCachedKernel>>>,
        >,
    >,
    topology: Option<Arc<themis::GpuTopology>>,
}

#[derive(Clone, Copy, Debug)]
struct CudaDeviceFeatures {
    shader_f64: bool,
    immediate_data: bool,
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
        Self::try_with_ordinal(0)
    }

    /// Acquire a CUDA device by ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::AdapterUnavailable`] when the CUDA driver or
    /// requested ordinal is unavailable. Returns
    /// [`HephaestusError::DeviceUnavailable`] when the context cannot be bound.
    pub fn try_with_ordinal(device_ordinal: usize) -> Result<Self> {
        let device_ordinal =
            i32::try_from(device_ordinal).map_err(|_| HephaestusError::AdapterUnavailable {
                message: format!("CUDA device ordinal {device_ordinal} exceeds i32 range"),
            })?;
        init_driver()?;
        let mut device = 0;
        // SAFETY: CUDA is initialized by `init_driver`; `device` is a valid
        // out-pointer for one driver device handle.
        let status = unsafe { cuda_oxide::sys::cuDeviceGet(&mut device, device_ordinal) };
        if status != 0 {
            return Err(HephaestusError::AdapterUnavailable {
                message: format!("CUDA device {device_ordinal} unavailable: {status}"),
            });
        }
        let context = Arc::new(CudaContext::create(device)?);
        context.bind()?;
        let limits = query_device_limits(&device)?;
        let features = query_device_features(&device)?;
        let topology = Some(Arc::new(query_topology(&device)?));
        let dev = Self {
            context,
            limits,
            features,
            pipeline_cache: Arc::new(moirai_sync::sync::ConcurrentHashMap::new()),
            topology,
        };
        // Sanity check to confirm that device allocation and copying are
        // functional (fails closed on headless/stub drivers).
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

    fn device_count() -> Result<usize> {
        init_driver()?;
        let mut count: core::ffi::c_int = 0;
        // SAFETY: `count` is a valid out-pointer for one `c_int`; the CUDA
        // driver has been initialized by `init_driver`.
        let status = unsafe { cuda_oxide::sys::cuDeviceGetCount(&mut count) };
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
                "max_immediate_size",
                u64::from(actual.max_immediate_size),
                u64::from(required.max_immediate_size),
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
        ) && available < needed
        {
            return Err(HephaestusError::DeviceUnavailable {
                message: format!(
                    "CUDA shader-stage storage-buffer limit {available} is below required {needed}"
                ),
            });
        }
        Ok(())
    }

    /// Resolve caller placement hints to CUDA's implemented allocation tier.
    ///
    /// This backend deliberately allocates primary buffers with `cuMemAlloc_v2`.
    /// That is non-managed device memory: host access happens only through
    /// explicit copies. Host-visible hints are therefore normalized to
    /// [`themis::MemoryTier::Device`] instead of being recorded as mappable or
    /// managed storage.
    fn allocation_tier(hint: themis::PlacementHint) -> Result<themis::MemoryTier> {
        match hint {
            themis::PlacementHint::Tier(tier) if !tier.is_host_allocatable() => {
                Err(HephaestusError::AllocationFailed {
                    message: format!(
                        "CUDA primary buffers cannot be allocated from budget-only tier {tier:?}"
                    ),
                })
            }
            themis::PlacementHint::Tier(_) | themis::PlacementHint::Current => {
                Ok(themis::MemoryTier::Device)
            }
            themis::PlacementHint::Numa(_)
            | themis::PlacementHint::Domain(_)
            | themis::PlacementHint::Any => Ok(themis::MemoryTier::Device),
        }
    }

    /// The device topology snapshot captured at acquisition, when available.
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.topology.as_deref()
    }

    /// The underlying cuda-oxide context handle (module lifetime management).
    #[inline]
    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Bind the device context to the current thread before a driver call.
    ///
    /// Transfers, allocations, module loads, and kernel launches execute
    /// against the thread's current context; binding makes this device's
    /// context current (CUDA contexts are thread-affine), so calls from any
    /// thread target the right device.
    pub fn bind(&self) -> Result<()> {
        self.context.bind()
    }

    /// Allocate `bytes` of device memory according to the tier.
    fn alloc_bytes(&self, bytes: usize) -> Result<DevicePtr> {
        self.bind()?;
        let byte_count = cuda_byte_count(bytes, "device allocation")?;
        let mut ptr = 0;
        // SAFETY: this device's context is current after `bind`; `ptr` is a
        // valid out-pointer for one `CUdeviceptr`, and `bytes > 0` at call
        // sites.
        let status = unsafe { cuda_oxide::sys::cuMemAlloc_v2(&mut ptr, byte_count) };
        if status != 0 {
            return Err(HephaestusError::AllocationFailed {
                message: format!("cuda-oxide cuMemAlloc_v2({bytes} bytes) -> {status}"),
            });
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
        let byte_count = cuda_byte_count(bytes, "download_sub_buffer byte count")?;
        let src_ptr = buffer.raw() + byte_offset;
        // SAFETY: `src_ptr` is a valid device pointer offset from a pointer allocated by this device;
        // `out` is `bytes` of writable host memory (`T: Pod`).
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, src_ptr, byte_count)
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
        let byte_count = cuda_byte_count(bytes, "write_sub_buffer byte count")?;
        let dest_ptr = buffer.raw() + byte_offset;
        // SAFETY: `dest_ptr` is a valid device pointer offset from a pointer allocated by this device;
        // `host` is `bytes` of readable host memory (`T: Pod`).
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyHtoD_v2(dest_ptr, host.as_ptr() as *const c_void, byte_count)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("write_sub_buffer cuMemcpyHtoD_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }
}

fn init_driver() -> Result<()> {
    Cuda::init().map_err(|e| HephaestusError::AdapterUnavailable {
        message: format!("cuda-oxide driver initialization failed: {e}"),
    })
}

fn device_attribute(
    device: cuda_oxide::sys::CUdevice,
    attribute: cuda_oxide::sys::CUdevice_attribute,
    name: &str,
) -> Result<i32> {
    let mut value: core::ffi::c_int = 0;
    // SAFETY: `device` is a valid `CUdevice` handle returned by cuda-oxide's
    // driver bindings; `value` is a valid out-pointer for one `c_int`.
    let result = unsafe { cuda_oxide::sys::cuDeviceGetAttribute(&mut value, attribute, device) };
    if result != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuDeviceGetAttribute({name}) -> {result}"),
        });
    }
    Ok(value)
}

fn current_memory_info() -> Result<(usize, usize)> {
    let mut free_bytes: cuda_oxide::sys::size_t = 0;
    let mut total_bytes: cuda_oxide::sys::size_t = 0;
    // SAFETY: the CUDA context is current for the calling thread at each call
    // site; both pointers address one writable `usize`.
    let result = unsafe { cuda_oxide::sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes) };
    if result != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuMemGetInfo_v2 -> {result}"),
        });
    }
    let free = usize::try_from(free_bytes).map_err(|_| HephaestusError::DeviceUnavailable {
        message: "CUDA free memory byte count exceeds usize".to_string(),
    })?;
    let total = usize::try_from(total_bytes).map_err(|_| HephaestusError::DeviceUnavailable {
        message: "CUDA total memory byte count exceeds usize".to_string(),
    })?;
    Ok((free, total))
}

pub(crate) fn cuda_byte_count(bytes: usize, what: &str) -> Result<cuda_oxide::sys::size_t> {
    cuda_oxide::sys::size_t::try_from(bytes).map_err(|_| HephaestusError::AllocationFailed {
        message: format!("{what} {bytes} exceeds cuda-oxide size_t range"),
    })
}

fn nonnegative_u32(value: i32) -> u32 {
    u32::try_from(value).unwrap_or(0)
}

fn query_device_limits(device: &cuda_oxide::sys::CUdevice) -> Result<DeviceLimits> {
    use cuda_oxide::sys;

    let (free_bytes, _) = current_memory_info()?;
    Ok(DeviceLimits {
        max_buffer_size: free_bytes as u64,
        max_compute_workgroup_size_x: nonnegative_u32(device_attribute(
            *device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            "max_block_dim_x",
        )?),
        max_compute_workgroup_size_y: nonnegative_u32(device_attribute(
            *device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            "max_block_dim_y",
        )?),
        max_compute_workgroup_size_z: nonnegative_u32(device_attribute(
            *device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            "max_block_dim_z",
        )?),
        max_compute_invocations_per_workgroup: nonnegative_u32(device_attribute(
            *device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            "max_threads_per_block",
        )?),
        max_compute_workgroup_storage_size: nonnegative_u32(device_attribute(
            *device,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            "max_shared_memory_per_block",
        )?),
        max_storage_buffers_per_shader_stage: None,
        max_buffers_and_acceleration_structures_per_shader_stage: None,
        max_immediate_size: 0,
    })
}

fn query_device_features(device: &cuda_oxide::sys::CUdevice) -> Result<CudaDeviceFeatures> {
    use cuda_oxide::sys;

    let major = device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        "compute_capability_major",
    )?;
    let minor = device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        "compute_capability_minor",
    )?;
    let compute_capability = major * 10 + minor;

    Ok(CudaDeviceFeatures {
        shader_f64: compute_capability >= 13,
        immediate_data: true,
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
fn query_topology(device: &cuda_oxide::sys::CUdevice) -> Result<themis::GpuTopology> {
    use cuda_oxide::sys;

    let compute_units = nonnegative_u32(device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        "multiprocessor_count",
    )?);
    let warp_width = nonnegative_u32(device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        "warp_size",
    )?);
    let max_threads_per_unit = nonnegative_u32(device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        "max_threads_per_multiprocessor",
    )?);
    let registers_per_unit = nonnegative_u32(device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        "max_registers_per_multiprocessor",
    )?);
    let shared_mem_per_unit_bytes = nonnegative_u32(device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        "max_shared_memory_per_multiprocessor",
    )?) as usize;
    let l2_bytes = nonnegative_u32(device_attribute(
        *device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        "l2_cache_size",
    )?) as usize;

    // Total device memory uses cuda-oxide's memory-management binding and the
    // context made current during acquisition.
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
            DeviceFeature::MappablePrimaryBuffers => false,
            DeviceFeature::ImmediateData => self.features.immediate_data,
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
        let tier = Self::allocation_tier(hint)?;
        if len == 0 {
            return Ok(CudaBuffer::new(0, 0, tier, self.context.clone()));
        }
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            HephaestusError::AllocationFailed {
                message: format!("byte count overflow for {len} elements"),
            }
        })?;
        let ptr = self.alloc_bytes(bytes)?;
        let byte_count = cuda_byte_count(bytes, "zero-init byte count")?;
        // SAFETY: `ptr` addresses `bytes` of device memory just allocated.
        let res = unsafe { cuda_oxide::sys::cuMemsetD8_v2(ptr, 0, byte_count) };
        let buffer = CudaBuffer::<T>::new(ptr, len, tier, self.context.clone());
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
        let tier = Self::allocation_tier(hint)?;
        if len == 0 {
            return Ok(CudaBuffer::new(0, 0, tier, self.context.clone()));
        }
        let bytes = core::mem::size_of_val(host);
        let ptr = self.alloc_bytes(bytes)?;
        let byte_count = cuda_byte_count(bytes, "upload byte count")?;
        // SAFETY: `ptr` addresses `bytes` of device memory just allocated;
        // `host` is `bytes` of readable host memory (`T: Pod`). The buffer owns
        // `ptr`, so it is freed if the copy fails.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyHtoD_v2(
                ptr,
                host.as_ptr().cast::<core::ffi::c_void>(),
                byte_count,
            )
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
        if buffer.len == 0 {
            return Ok(());
        }
        let bytes = core::mem::size_of_val(out);
        let byte_count = cuda_byte_count(bytes, "download byte count")?;
        self.bind()?;
        // SAFETY: `buffer.ptr` addresses `bytes` of device memory (len matches,
        // checked above); `out` is `bytes` of writable host memory (`T: Pod`).
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr().cast::<c_void>(),
                buffer.ptr,
                byte_count,
            )
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
        let byte_count = cuda_byte_count(bytes, "write_buffer byte count")?;
        // SAFETY: `buffer.ptr` is a valid device pointer allocated by this
        // device; `host` is `bytes` of readable host memory (`T: Pod`).
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyHtoD_v2(
                buffer.raw(),
                host.as_ptr() as *const c_void,
                byte_count,
            )
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
        let res = unsafe { cuda_oxide::sys::cuCtxSynchronize() };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("cuCtxSynchronize -> {res}"),
            });
        }
        Ok(())
    }
}
