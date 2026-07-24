use hephaestus_core::{DeviceLimits, HephaestusError, Result};

use super::device::{HIP_SUCCESS, RocmContext, check_status};

fn query_attribute(
    device: i32,
    attribute: cubecl_hip_sys::hipDeviceAttribute_t,
    name: &str,
) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a valid out-pointer and `device`/`attribute` are
    // supplied by HIP's acquired-device contract.
    let status = unsafe { cubecl_hip_sys::hipDeviceGetAttribute(&mut value, attribute, device) };
    if status == HIP_SUCCESS {
        Ok(value)
    } else {
        Err(HephaestusError::DeviceUnavailable {
            message: super::device::status_message(status, name),
        })
    }
}

fn positive_u32(value: i32, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DeviceUnavailable {
        message: format!("ROCm attribute {name} returned negative value {value}"),
    })
}

pub(super) fn query_device_limits(context: &RocmContext) -> Result<DeviceLimits> {
    context.set_current()?;
    let mut free_bytes = 0;
    let mut total_bytes = 0;
    // SAFETY: both locals are valid out-pointers for HIP's byte counts.
    let status = unsafe { cubecl_hip_sys::hipMemGetInfo(&mut free_bytes, &mut total_bytes) };
    check_status(status, "hipMemGetInfo")?;
    let ordinal = context.ordinal();

    Ok(DeviceLimits {
        max_buffer_size: u64::try_from(free_bytes).map_err(|_| {
            HephaestusError::DeviceUnavailable {
                message: format!("ROCm free memory {free_bytes} exceeds u64"),
            }
        })?,
        max_compute_workgroup_size_x: positive_u32(
            query_attribute(
                ordinal,
                cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimX,
                "max_block_dim_x",
            )?,
            "max_block_dim_x",
        )?,
        max_compute_workgroup_size_y: positive_u32(
            query_attribute(
                ordinal,
                cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimY,
                "max_block_dim_y",
            )?,
            "max_block_dim_y",
        )?,
        max_compute_workgroup_size_z: positive_u32(
            query_attribute(
                ordinal,
                cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimZ,
                "max_block_dim_z",
            )?,
            "max_block_dim_z",
        )?,
        max_compute_invocations_per_workgroup: positive_u32(
            query_attribute(
                ordinal,
                cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerBlock,
                "max_threads_per_block",
            )?,
            "max_threads_per_block",
        )?,
        max_compute_workgroup_storage_size: positive_u32(
            query_attribute(
                ordinal,
                cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerBlock,
                "max_shared_memory_per_block",
            )?,
            "max_shared_memory_per_block",
        )?,
        max_storage_buffers_per_shader_stage: None,
        max_buffers_and_acceleration_structures_per_shader_stage: None,
        max_immediate_size: 0,
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RocmDeviceFeatures {
    pub(super) mappable_primary_buffers: bool,
}

pub(super) fn query_device_features() -> RocmDeviceFeatures {
    // `hipMalloc` allocations are explicit device memory. Supporting
    // host-mapped allocations would require a distinct allocation tier and
    // transfer contract, so the primary-buffer capability stays disabled.
    RocmDeviceFeatures {
        mappable_primary_buffers: false,
    }
}

pub(super) fn query_topology(context: &RocmContext) -> Result<themis::GpuTopology> {
    context.set_current()?;
    let device = context.ordinal();
    let compute_units = positive_u32(
        query_attribute(
            device,
            cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount,
            "multiprocessor_count",
        )?,
        "multiprocessor_count",
    )?;
    let warp_width = positive_u32(
        query_attribute(
            device,
            cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeWarpSize,
            "warp_size",
        )?,
        "warp_size",
    )?;
    let max_threads_per_unit = positive_u32(
        query_attribute(
            device,
            cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerMultiProcessor,
            "max_threads_per_multiprocessor",
        )?,
        "max_threads_per_multiprocessor",
    )?;
    let registers_per_unit = positive_u32(
        query_attribute(
            device,
            cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeMaxRegistersPerMultiprocessor,
            "max_registers_per_multiprocessor",
        )?,
        "max_registers_per_multiprocessor",
    )?;
    let shared_mem_per_unit_bytes = usize::try_from(query_attribute(
        device,
        cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeSharedMemPerMultiprocessor,
        "shared_memory_per_multiprocessor",
    )?)
    .map_err(|_| HephaestusError::DeviceUnavailable {
        message: "ROCm shared memory per multiprocessor exceeds usize".to_string(),
    })?;
    let l2_bytes = usize::try_from(query_attribute(
        device,
        cubecl_hip_sys::hipDeviceAttribute_t_hipDeviceAttributeL2CacheSize,
        "l2_cache_size",
    )?)
    .map_err(|_| HephaestusError::DeviceUnavailable {
        message: "ROCm L2 cache size exceeds usize".to_string(),
    })?;
    let mut free_bytes = 0;
    let mut total_bytes = 0;
    // SAFETY: both locals are valid out-pointers for HIP's byte counts.
    let status = unsafe { cubecl_hip_sys::hipMemGetInfo(&mut free_bytes, &mut total_bytes) };
    check_status(status, "hipMemGetInfo")?;

    Ok(themis::GpuTopology::from_provider(
        themis::GpuDeviceProperties {
            compute_units,
            warp_width,
            max_threads_per_unit,
            registers_per_unit,
            shared_mem_per_unit_bytes,
            l2_bytes,
            memory_tier: themis::MemoryTier::Device,
            memory_bytes: u64::try_from(total_bytes).map_err(|_| {
                HephaestusError::DeviceUnavailable {
                    message: format!("ROCm total memory {total_bytes} exceeds u64"),
                }
            })?,
        },
    ))
}
