//! Device substrate: typed device buffer + `ComputeDevice` acquisition and
//! transfer. The real implementations require the `cuda` feature; without it
//! the stub variants compile unsafe-free and report the backend unavailable.

#[cfg(feature = "cuda")]
pub mod buffer;
#[cfg(not(feature = "cuda"))]
#[path = "buffer_stub.rs"]
pub mod buffer;

#[cfg(feature = "cuda")]
pub mod device;
#[cfg(not(feature = "cuda"))]
#[path = "device_stub.rs"]
pub mod device;

#[cfg(feature = "cuda")]
pub mod compiler;

#[cfg(all(feature = "cuda", feature = "decomposition"))]
pub(crate) mod pinned;
