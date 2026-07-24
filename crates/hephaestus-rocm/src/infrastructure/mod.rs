#[cfg(all(feature = "rocm", target_os = "linux"))]
mod acquisition;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod buffer;
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
mod buffer_stub;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod capabilities;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod device;
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
#[cfg(all(feature = "rocm", target_os = "linux"))]
pub use device::RocmDevice;
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
pub use device_stub::RocmDevice;
