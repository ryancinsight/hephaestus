//! CUDA connectionist temporal classification.
//!
//! The CUDA counterpart of the WGSL forward-backward loss, computing the
//! same compensated log-space recurrences from the same problem buffers.
//! One CUDA source serves all four entries — the alpha and beta frame
//! steps, the likelihood finish, and the posterior sweep — and the
//! per-lane arithmetic keeps the provider's exact operation order. `f32`
//! only, matching the uniform block layout the steps share with WGSL.
//!
//! # Bit-exactness against the CPU path, up to transcendentals
//!
//! NVRTC compiles without fast-math, and the kernels hold no
//! multiply-add pair for contraction to fuse — the TwoSum chains add and
//! subtract only, and the posterior's lone product feeds divisions — so
//! the same rational operations round the same way as on the host. Only
//! `expf` and `logf` may differ from the host libm by a unit in the last
//! place. Infinities are spelled as bit patterns through
//! `__int_as_float`, so the unreachable checks are exact on every
//! architecture.

mod kernel;
mod ops;
mod source;

pub use kernel::CtcKernel;
pub use ops::CudaCtcOps;
