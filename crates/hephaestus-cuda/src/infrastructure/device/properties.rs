use super::state::CudaDeviceFeatures;
use crate::infrastructure::driver::{Attribute, Driver};
use hephaestus_core::{DeviceLimits, HephaestusError, Result};

pub(super) fn device_attribute(
    driver: &Driver,
    device: i32,
    attribute: Attribute,
    name: &str,
) -> Result<i32> {
    let mut value: core::ffi::c_int = 0;
    // SAFETY: `device` is a valid `CUdevice` handle returned by the CUDA driver's
    // driver bindings; `value` is a valid out-pointer for one `c_int`.
    let result = unsafe { (driver.device.attribute)(&mut value, attribute, device) };
    if result != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuDeviceGetAttribute({name}) -> {result}"),
        });
    }
    if value < 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("CUDA device attribute {name} is negative: {value}"),
        });
    }
    Ok(value)
}

pub(super) fn current_memory_info(driver: &Driver) -> Result<(usize, usize)> {
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    // SAFETY: the CUDA context is current for the calling thread at each call
    // site; both pointers address one writable `usize`.
    let result = unsafe { (driver.memory.info)(&mut free_bytes, &mut total_bytes) };
    if result != 0 {
        return Err(HephaestusError::DeviceUnavailable {
            message: format!("cuMemGetInfo_v2 -> {result}"),
        });
    }
    Ok((free_bytes, total_bytes))
}

fn nonnegative_u32(value: i32) -> u32 {
    u32::try_from(value).expect("invariant: device_attribute rejects negative values")
}

pub(super) fn query_device_limits(driver: &Driver, device: &i32) -> Result<DeviceLimits> {
    // The limit is the device's capacity, a stable per-device fact like the
    // workgroup bounds below; the free-memory snapshot it used to carry went
    // stale after the first allocation and made `require_limits` reject busy
    // devices. Free memory is a runtime query (`CudaDevice::free_memory_bytes`).
    let (_, total_bytes) = current_memory_info(driver)?;
    let max_buffer_size =
        u64::try_from(total_bytes).map_err(|_| HephaestusError::DeviceUnavailable {
            message: "CUDA total memory byte count exceeds u64".to_string(),
        })?;
    Ok(DeviceLimits {
        max_buffer_size,
        max_compute_workgroup_size_x: nonnegative_u32(device_attribute(
            driver,
            *device,
            Attribute::MaxBlockDimX,
            "max_block_dim_x",
        )?),
        max_compute_workgroup_size_y: nonnegative_u32(device_attribute(
            driver,
            *device,
            Attribute::MaxBlockDimY,
            "max_block_dim_y",
        )?),
        max_compute_workgroup_size_z: nonnegative_u32(device_attribute(
            driver,
            *device,
            Attribute::MaxBlockDimZ,
            "max_block_dim_z",
        )?),
        max_compute_invocations_per_workgroup: nonnegative_u32(device_attribute(
            driver,
            *device,
            Attribute::MaxThreadsPerBlock,
            "max_threads_per_block",
        )?),
        max_compute_workgroup_storage_size: nonnegative_u32(device_attribute(
            driver,
            *device,
            Attribute::MaxSharedMemoryPerBlock,
            "max_shared_memory_per_block",
        )?),
        max_storage_buffers_per_shader_stage: None,
        max_buffers_and_acceleration_structures_per_shader_stage: None,
        max_immediate_size: 0,
    })
}

pub(super) fn query_device_features(driver: &Driver, device: &i32) -> Result<CudaDeviceFeatures> {
    let major = device_attribute(
        driver,
        *device,
        Attribute::ComputeCapabilityMajor,
        "compute_capability_major",
    )?;
    let minor = device_attribute(
        driver,
        *device,
        Attribute::ComputeCapabilityMinor,
        "compute_capability_minor",
    )?;
    let compute_capability = major * 10 + minor;

    Ok(CudaDeviceFeatures {
        compute_capability,
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
pub(super) fn query_topology(driver: &Driver, device: &i32) -> Result<themis::GpuTopology> {
    let compute_units = nonnegative_u32(device_attribute(
        driver,
        *device,
        Attribute::MultiprocessorCount,
        "multiprocessor_count",
    )?);
    let warp_width = nonnegative_u32(device_attribute(
        driver,
        *device,
        Attribute::WarpSize,
        "warp_size",
    )?);
    let max_threads_per_unit = nonnegative_u32(device_attribute(
        driver,
        *device,
        Attribute::MaxThreadsPerMultiprocessor,
        "max_threads_per_multiprocessor",
    )?);
    let registers_per_unit = nonnegative_u32(device_attribute(
        driver,
        *device,
        Attribute::MaxRegistersPerMultiprocessor,
        "max_registers_per_multiprocessor",
    )?);
    let shared_mem_per_unit_bytes = nonnegative_u32(device_attribute(
        driver,
        *device,
        Attribute::MaxSharedMemoryPerMultiprocessor,
        "max_shared_memory_per_multiprocessor",
    )?) as usize;
    let l2_bytes = nonnegative_u32(device_attribute(
        driver,
        *device,
        Attribute::L2CacheSize,
        "l2_cache_size",
    )?) as usize;

    // Total device memory uses the driver table and the
    // context made current during acquisition.
    let (_, total_bytes) = current_memory_info(driver)?;

    Ok(themis::GpuTopology::from_provider(
        themis::GpuDeviceProperties {
            compute_units: core::num::NonZeroU32::new(compute_units),
            warp_width: core::num::NonZeroU32::new(warp_width),
            max_threads_per_unit: core::num::NonZeroU32::new(max_threads_per_unit),
            registers_per_unit: core::num::NonZeroU32::new(registers_per_unit),
            shared_mem_per_unit_bytes: core::num::NonZeroUsize::new(shared_mem_per_unit_bytes),
            l2_bytes: core::num::NonZeroUsize::new(l2_bytes),
            memory_tier: themis::MemoryTier::Device,
            memory_bytes: core::num::NonZeroU64::new(total_bytes as u64),
        },
    ))
}
