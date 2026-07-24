#![cfg_attr(not(all(feature = "rocm", target_os = "linux")), forbid(unsafe_code))]
#![deny(missing_docs)]

//! # hephaestus-rocm
//!
//! Native AMD ROCm/HIP device substrate for the Atlas accelerator stack.
//!
//! The `rocm` feature enables the Linux HIP runtime implementation. Without
//! that feature, [`RocmDevice::try_default`] returns a typed unavailable-device
//! error and the crate remains buildable on hosts without ROCm. The backend
//! implements the shared [`hephaestus_core::ComputeDevice`] seam for device
//! acquisition, typed device buffers, host/device transfers, and
//! synchronization. HIP kernel authoring is a separate application-layer
//! increment.
//!
//! [`hephaestus_core::ComputeDevice`]: hephaestus_core::ComputeDevice

#[cfg(all(feature = "rocm", not(target_os = "linux")))]
compile_error!("the hephaestus-rocm `rocm` feature requires a Linux ROCm installation");

mod infrastructure;

pub use infrastructure::{RocmBuffer, RocmDevice};

pub use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceBuffer,
    DeviceFeature, DeviceLimits, DevicePreference, HephaestusError, Result,
};
