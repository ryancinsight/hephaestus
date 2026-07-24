use std::ffi::CStr;
use std::sync::Arc;

use hephaestus_core::{ComputeDevice, DeviceLimits, HephaestusError, Result};

use super::query::{
    RocmDeviceFeatures, query_device_features, query_device_limits, query_topology,
};

pub(super) const HIP_SUCCESS: cubecl_hip_sys::hipError_t = cubecl_hip_sys::HIP_SUCCESS;

/// A thread-bindable ROCm device identity.
///
/// HIP stores the current device in thread-local state. The context records
/// the ordinal and rebinds it before every runtime operation, including drop.
#[derive(Debug)]
pub(crate) struct RocmContext {
    ordinal: i32,
}

impl RocmContext {
    fn new(ordinal: i32) -> Self {
        Self { ordinal }
    }

    pub(super) fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub(crate) fn set_current(&self) -> Result<()> {
        // SAFETY: `self.ordinal` was validated against `hipGetDeviceCount`
        // during acquisition and is passed unchanged to HIP's thread-local
        // device selector.
        let status = unsafe { cubecl_hip_sys::hipSetDevice(self.ordinal) };
        check_status(status, "hipSetDevice")
    }
}

pub(super) fn status_message(status: cubecl_hip_sys::hipError_t, operation: &str) -> String {
    // SAFETY: HIP returns either a null pointer or a process-lifetime
    // null-terminated diagnostic string for an error code.
    let detail = unsafe {
        let ptr = cubecl_hip_sys::hipGetErrorString(status);
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    };
    match detail {
        Some(detail) => format!("{operation} -> {detail} (status {status})"),
        None => format!("{operation} -> unknown HIP status {status}"),
    }
}

pub(crate) fn check_status(status: cubecl_hip_sys::hipError_t, operation: &str) -> Result<()> {
    if status == HIP_SUCCESS {
        Ok(())
    } else {
        Err(HephaestusError::DeviceUnavailable {
            message: status_message(status, operation),
        })
    }
}

/// An acquired AMD device backed by the ROCm HIP runtime.
#[derive(Clone)]
pub struct RocmDevice {
    pub(super) context: Arc<RocmContext>,
    pub(super) limits: DeviceLimits,
    pub(super) features: RocmDeviceFeatures,
    topology: Arc<themis::GpuTopology>,
}

impl core::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RocmDevice").finish_non_exhaustive()
    }
}

impl RocmDevice {
    /// Acquire the default AMD device (ordinal zero).
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::AdapterUnavailable`] when ROCm or an AMD
    /// device is unavailable, and a typed device error when HIP cannot query
    /// the acquired device.
    pub fn try_default() -> Result<Self> {
        Self::try_with_ordinal(0)
    }

    /// Acquire an AMD device by HIP ordinal.
    ///
    /// # Errors
    ///
    /// Returns a typed acquisition error when the HIP runtime or ordinal is
    /// unavailable, or when the acquired device cannot satisfy its own
    /// driver-backed capability queries.
    pub fn try_with_ordinal(device_ordinal: usize) -> Result<Self> {
        let ordinal =
            i32::try_from(device_ordinal).map_err(|_| HephaestusError::AdapterUnavailable {
                message: format!("ROCm device ordinal {device_ordinal} exceeds i32 range"),
            })?;
        let mut count = 0;
        // SAFETY: `count` is a valid out-pointer for HIP's device count.
        let status = unsafe { cubecl_hip_sys::hipGetDeviceCount(&mut count) };
        if status != HIP_SUCCESS {
            return Err(HephaestusError::AdapterUnavailable {
                message: status_message(status, "hipGetDeviceCount"),
            });
        }
        if ordinal < 0 || ordinal >= count {
            return Err(HephaestusError::AdapterUnavailable {
                message: format!("ROCm device ordinal {ordinal} unavailable; device count {count}"),
            });
        }

        let context = Arc::new(RocmContext::new(ordinal));
        context.set_current()?;
        let limits = query_device_limits(&context)?;
        let features = query_device_features();
        let topology = Arc::new(query_topology(&context)?);
        let device = Self {
            context,
            limits,
            features,
            topology,
        };

        let buffer =
            device
                .alloc_zeroed::<u32>(1)
                .map_err(|error| HephaestusError::AdapterUnavailable {
                    message: format!("ROCm sanity allocation failed: {error}"),
                })?;
        let host = [42_u32];
        device.write_buffer(&buffer, &host).map_err(|error| {
            HephaestusError::AdapterUnavailable {
                message: format!("ROCm sanity write failed: {error}"),
            }
        })?;
        let mut roundtrip = [0_u32];
        device.download(&buffer, &mut roundtrip).map_err(|error| {
            HephaestusError::AdapterUnavailable {
                message: format!("ROCm sanity download failed: {error}"),
            }
        })?;
        if roundtrip != host {
            return Err(HephaestusError::AdapterUnavailable {
                message: format!("ROCm sanity transfer mismatch: {roundtrip:?} != {host:?}"),
            });
        }
        Ok(device)
    }

    /// Return the device's provider topology snapshot.
    #[must_use]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        Some(&self.topology)
    }

    pub(super) fn allocation_tier(hint: themis::PlacementHint) -> Result<themis::MemoryTier> {
        match hint {
            themis::PlacementHint::Tier(tier) if !tier.is_host_allocatable() => {
                Err(HephaestusError::AllocationFailed {
                    message: format!(
                        "ROCm primary buffers cannot be allocated from budget-only tier {tier:?}"
                    ),
                })
            }
            themis::PlacementHint::Tier(_)
            | themis::PlacementHint::Current
            | themis::PlacementHint::Numa(_)
            | themis::PlacementHint::Domain(_)
            | themis::PlacementHint::Any => Ok(themis::MemoryTier::Device),
        }
    }

    pub(super) fn require_limits(actual: DeviceLimits, required: DeviceLimits) -> Result<()> {
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
                        "ROCm device limit {name} {available} is below required {needed}"
                    ),
                });
            }
        }
        Ok(())
    }

    pub(super) fn device_count() -> Result<usize> {
        let mut count = 0;
        // SAFETY: `count` is a valid out-pointer for HIP's device count.
        let status = unsafe { cubecl_hip_sys::hipGetDeviceCount(&mut count) };
        if status != HIP_SUCCESS {
            return Err(HephaestusError::AdapterUnavailable {
                message: status_message(status, "hipGetDeviceCount"),
            });
        }
        usize::try_from(count).map_err(|_| HephaestusError::AdapterUnavailable {
            message: format!("ROCm device count {count} is negative"),
        })
    }
}
