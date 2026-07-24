#[cfg(all(feature = "rocm", target_os = "linux"))]
mod acquisition;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod buffer;
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
mod buffer_stub;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod capabilities;
#[cfg(all(feature = "rocm", target_os = "linux"))]
pub(crate) mod device;
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
mod device_stub;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod memory;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod query;

#[cfg(all(feature = "rocm", target_os = "linux"))]
pub use buffer::RocmBuffer;
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
pub use buffer_stub::RocmBuffer;
use core::ffi::c_void;
#[cfg(all(feature = "rocm", target_os = "linux"))]
pub use device::RocmDevice;
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
pub use device_stub::RocmDevice;

/// Opaque HIP device address used by ROCm kernel-launch code.
pub(crate) type DevicePtr = *mut c_void;
