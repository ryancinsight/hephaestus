// The stub substrate (no `cuda` feature) performs no FFI and forbids unsafe.
// The real backend requires unsafe for the dynamic-loaded driver FFI; it is
// isolated to `infrastructure::{device, buffer}` with per-block SAFETY notes.
#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![deny(missing_docs)]
//! # hephaestus-cuda
//!
//! CUDA backend for the Atlas shared accelerator substrate (atlas ADR 0001).
//! It is the GPU-side sibling of [`hephaestus-wgpu`]: it implements the same
//! [`hephaestus_core::ComputeDevice`] seam — device acquisition, a typed
//! [`CudaBuffer<T>`] device buffer, and host/device transfer — so consumers
//! that bind generically (`<D: ComputeDevice>`) substitute CUDA for wgpu
//! without source changes. Hephaestus is to the GPU what `leto` is to the CPU;
//! this crate adds the CUDA pathway alongside the portable wgpu one.
//!
//! The CUDA toolchain is composed from cutile-rs (`cuda-core` driver `sys`
//! bindings + `cuda-async` device/context acquisition), the same source
//! `coeus-cuda` uses. It is gated behind the `cuda` feature and loaded
//! dynamically at runtime (`nvcuda.dll` / `libcuda.so`), so building does not
//! require a CUDA toolkit. Without the feature, the crate compiles as a stub
//! whose [`CudaDevice::try_default`] reports the backend unavailable rather
//! than fabricating a device.
//!
//! [`hephaestus-wgpu`]: https://docs.rs/hephaestus-wgpu
//! [`CudaBuffer<T>`]: crate::CudaBuffer
//! [`CudaDevice::try_default`]: crate::CudaDevice::try_default
//!
//! ## Scope
//!
//! This crate currently owns the device substrate (acquisition, typed buffers,
//! transfer). Monomorphized elementwise/reduction kernel dispatch — mirroring
//! `hephaestus-wgpu`'s `application` layer — composes cutile PTX authoring on
//! top of this foundation and lands in a follow-up.

mod infrastructure;

/// Monomorphized CUDA compute dispatch.
pub mod application;

pub use application::cuda_type::CudaScalar;
pub use application::elementwise::{
    binary_elementwise, binary_elementwise_into, scalar_elementwise, scalar_elementwise_into,
    unary_elementwise, unary_elementwise_into, AbsOp, AddOp, BinaryCudaOp, CosOp, DivOp, ExpOp,
    IdentityOp, LnOp, MulOp, NegOp, PowOp, RecipOp, SinOp, SqrtOp, SubOp, UnaryCudaOp,
};
pub use application::linalg::{batched_matmul, dot, matmul, norm_l1, norm_l2, norm_max, trace};
pub use application::reduction::{
    reduction, reduction_with_width, MaxOp, MinOp, ReductionCudaOp, ReductionIdentity, SumOp,
};
pub use application::strided::{
    binary_elementwise_strided_into, scalar_elementwise_strided_into,
    unary_elementwise_strided_into, StridedOperand, MAX_STRIDED_RANK,
};

pub use infrastructure::buffer::CudaBuffer;
pub use infrastructure::device::CudaDevice;

pub use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};
