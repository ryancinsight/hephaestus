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
    binary_elementwise, scalar_elementwise, unary_elementwise, AddOp, BinaryWgslOp, MulOp, SubOp,
    AbsOp, CosOp, ExpOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp, UnaryWgslOp,
};
pub use application::wgsl::WgslScalar;
pub use infrastructure::buffer::WgpuBuffer;
pub use infrastructure::device::WgpuDevice;

pub use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};
