#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! Backend-neutral axis-reduction planning and the device-neutral seam.
//!
//! Both GPU backends reduce a rank-2 strided operand along one axis with the
//! identical host-side contract: validate the shapes/strides, pack the launch
//! metadata into a `#[repr(C)]` block mirroring the shader's uniform struct,
//! and derive the workgroup count from the output length. That logic is
//! dialect-independent — only the shader source and raw dispatch differ — so
//! it lives here once and both backends call [`plan_axis_reduction`](crate::plan_axis_reduction).
//!
//! The one backend-specific input is whether the input and output buffers
//! alias (a device-pointer identity check); callers compute it and pass it in.
//!
//! [`crate::AxisReductionOps`] completes the family: the planning above was already
//! shared, but every backend still exposed `prod_axis_into` and
//! `prepare_reduce_axis_into` as free functions over its own device and operand
//! types, so a consumer — or a conformance suite — had to bind to one device API
//! to call them. The seam takes the same shape as [`crate::DenseVectorOps`] and
//! [`crate::SparseOperatorOps`]: generic over the device, monomorphized at every call
//! site, no `dyn` anywhere.

mod planning;
mod seams;
#[cfg(test)]
mod tests;

pub use planning::{
    AxisReductionDispatch, AxisReductionMeta, plan_axis_reduction, reduction_pass_count,
    validate_reduction_width,
};
pub use seams::{AxisReductionOps, FullReductionOps, StridedComputeBackend};
