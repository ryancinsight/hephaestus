#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Differential contract tests for CUDA strided-layout dispatch: device results vs
//! a CPU reference computed over the same leto layout metadata.

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_cuda::{
    AddOp, CudaBuffer, CudaDevice, EqOp, MulOp, NegOp, SqrtOp, StridedLayout, StridedOperand,
    StridedOperandDyn, binary_elementwise_strided, binary_elementwise_strided_dyn_into,
    binary_elementwise_strided_into, binary_elementwise_strided_typed, scalar_elementwise_strided,
    scalar_elementwise_strided_into, unary_elementwise_strided, unary_elementwise_strided_dyn_into,
    unary_elementwise_strided_into,
};
use leto::Layout;

#[path = "strided/fixtures.rs"]
mod fixtures;
#[path = "strided/strided_views.rs"]
mod strided_views;
pub(crate) use fixtures::*;
