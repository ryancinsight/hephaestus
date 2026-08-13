//! Atlas accelerator substrate — the entry point for the Hephaestus workspace.
//!
//! This crate is a facade. It owns no logic: it re-exports the device, buffer,
//! transfer, and kernel contracts from `hephaestus-core`, and each backend
//! implementation behind a feature. Depend on `hephaestus`; the sub-crates exist
//! so a consumer *can* take a narrower dependency, not because it should.
//!
//! # Layers
//!
//! The contract layer is always present and pulls in no device stack, so code
//! generic over a backend compiles anywhere:
//!
//! ```
//! use hephaestus::DeviceBuffer;
//!
//! // Generic over element type and backend. Naming the contract requires no
//! // backend feature, so this compiles on a machine with no accelerator at all.
//! fn total_elements<T, B: DeviceBuffer<T>>(buffers: &[B]) -> usize {
//!     buffers.iter().map(|buffer| buffer.len()).sum()
//! }
//! ```
//!
//! Backends are opt-in, each under a module named for its device API:
//!
//! ```toml
//! # portable compute, no vendor toolkit needed
//! hephaestus = { version = "0.19", features = ["wgpu"] }
//!
//! # NVIDIA; needs a CUDA toolkit at build time for headers
//! hephaestus = { version = "0.19", features = ["cuda"] }
//! ```
//!
//! No backend is enabled by default. A default backend would make every
//! consumer of the traits pull a device stack, and `cuda`/`rocm` additionally
//! require vendor toolkits present at build time.
//!
//! # Feature flags select what is compiled
//!
//! Enabling a backend feature compiles it in. This facade performs no backend
//! selection: it contains no logic, only re-exports. Select a backend
//! explicitly by naming its device type (`WgpuDevice::try_default`,
//! `CudaDevice::try_default`, …); each returns a typed unavailable-device error
//! on a host without that hardware, so a caller that wants a preference order
//! writes that order itself.
//!
//! | Feature | Effect |
//! | --- | --- |
//! | `wgpu` | portable compute backend; the usual starting point |
//! | `cuda` | NVIDIA backend; CUDA toolkit needed at build time |
//! | `rocm` | AMD backend; HIP system libraries needed at build time |
//! | `metal` | Apple backend |
//! | `decomposition` | dense Cholesky/LU/QR on the enabled backends |
//! | `sparse` | CSR upload/download and GPU sparse products |
//! | `parallel`, `mnemosyne-memory` | forwarded to the contract layer and every enabled backend |

// Deliberately not `no_std`: `hephaestus-core` is a std crate, so declaring the
// facade `no_std` would advertise a portability the re-exported surface does not
// have.
#![deny(missing_docs)]

// The contract layer is re-exported flat so items appear at facade paths
// (`hephaestus::DeviceBuffer`) rather than sending readers into a sub-crate.
#[doc(inline)]
pub use hephaestus_core::*;

/// Contract layer under its own path, for code that wants to name it explicitly.
///
/// `hephaestus::contracts::DeviceBuffer` and `hephaestus::DeviceBuffer` are the
/// same item; the alias exists so a caller can disambiguate against a
/// backend-specific type of the same name.
pub use hephaestus_core as contracts;

/// Portable compute backend over `wgpu`.
#[cfg(feature = "wgpu")]
pub use hephaestus_wgpu as wgpu;

/// NVIDIA backend.
#[cfg(feature = "cuda")]
pub use hephaestus_cuda as cuda;

/// AMD backend over HIP.
#[cfg(feature = "rocm")]
pub use hephaestus_rocm as rocm;

/// Apple backend.
#[cfg(feature = "metal")]
pub use hephaestus_metal as metal;
