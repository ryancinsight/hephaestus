#![deny(missing_docs)]
//! # hephaestus-wgpu
//!
//! The portable wgpu backend of the Atlas accelerator substrate (atlas ADR
//! 0001). Implements the `hephaestus-core` [`ComputeDevice`] seam over a
//! wgpu device/queue pair: adapter acquisition, typed device buffers, and
//! monomorphized elementwise compute dispatch.
//!
//! [`ComputeDevice`]: hephaestus_core::ComputeDevice

/// Elementwise compute dispatch.
pub mod application;
/// wgpu device, queue, and buffer infrastructure.
pub mod infrastructure;

pub use application::elementwise::{
    binary_elementwise, binary_elementwise_into, scalar_elementwise, scalar_elementwise_into,
    unary_elementwise, unary_elementwise_into, AbsOp, AddOp, BinaryWgslOp, CosOp, DivOp, ExpOp,
    LnOp, MulOp, NegOp, PowOp, RecipOp, SinOp, SqrtOp, SubOp, UnaryWgslOp,
};
pub use application::reduction::{
    reduction, reduction_with_width, MaxOp, MinOp, ReductionIdentity, ReductionWgslOp, SumOp,
};
pub use application::strided::{
    binary_elementwise_strided_into, scalar_elementwise_strided_into,
    unary_elementwise_strided_into, StridedOperand, MAX_STRIDED_RANK,
};
pub use application::wgsl::WgslScalar;
pub use infrastructure::buffer::WgpuBuffer;
pub use infrastructure::device::WgpuDevice;

pub use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};
