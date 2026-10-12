#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Contract tests for the Metal `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without macOS or without a Metal device, [`MetalDevice::try_default`]
//! returns `Err` and each test skips. Hardware CI sets
//! `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` so an unavailable device fails that lane
//! instead of being reported as device evidence.

use std::borrow::Cow;

use hephaestus_core::{
    BinaryStorageKernel, Binding, BindingDecl, BlockWidth, CommandStream, ComputeDevice,
    ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature, DeviceLimits,
    DevicePreference, DispatchGrid, GroupedBinding, GroupedBindingDecl, GroupedCommandStream,
    GroupedKernelDevice, GroupedKernelInterface, GroupedKernelSequence, GroupedKernelSource,
    HephaestusError, KernelDevice, KernelInterface, KernelSource, MultiStorageKernel, Result,
    UnaryStorageKernel, Wgsl,
};
use hephaestus_metal::{
    AddOp, EluGradOp, EluOp, ExpNegOp, ExpOp, GeluTanhGradOp, GeluTanhOp, MatrixFunction,
    MatrixNorm, MatrixProduct, MatrixProperties, MatrixSolve, MaxOp, MetalBinaryStorageKernel,
    MetalDevice, MetalMultiStorageKernel, MetalStorageBinding, MetalStorageBindingLayout,
    MetalUnaryStorageKernel, MinOp, MishGradOp, MishOp, MulOp, NegOp, ProdOp, SiluGradOp, SiluOp,
    SoftplusGradOp, SoftplusOp, SqrtOp, StridedOperand, SumOp, binary_elementwise, cumprod,
    cumprod_into, matmul, normal_with_seed, prepare_dot, prepare_max_axis_into,
    prepare_mean_axis_into, prepare_min_axis_into, prepare_norm_l2, prepare_reduction,
    prepare_reduction_with_width, prepare_sum_axis_into, prod_axis, prod_axis_into,
    reduce_axis_into, reduction, scalar_elementwise, submit_prepared_axis_reduction_batch,
    submit_prepared_reduction_batch, suffix_prod, suffix_prod_into, suffix_sum, suffix_sum_into,
    unary_elementwise, unary_elementwise_into, uniform_with_seed,
};
#[cfg(feature = "decomposition")]
use hephaestus_metal::{
    MatrixDecompose, col_piv_qr, col_piv_qr_blocked, full_piv_lu, full_piv_lu_blocked,
};
use hephaestus_wgpu::wgpu;
use leto::Layout;

impl KernelInterface for StreamScaleKernel {
    type Params = StreamParams;

    const LABEL: &'static str = "hephaestus-metal-stream-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl GroupedKernelInterface for GroupedAddKernel {
    type Params = GroupedParams;

    const LABEL: &'static str = "hephaestus-metal-grouped-add";
    const BINDINGS: &'static [GroupedBindingDecl] = &[
        GroupedBindingDecl::read_only::<f32>(0, 0),
        GroupedBindingDecl::read_only::<f32>(1, 0),
        GroupedBindingDecl::read_write::<f32>(1, 1),
    ];
    const PARAM_GROUP: u32 = 0;
    const PARAM_BINDING: u32 = 1;
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

#[path = "contract/axis.rs"]
mod axis;
#[path = "contract/buffer.rs"]
mod buffer;
#[path = "contract/device.rs"]
mod device;
#[path = "contract/elementwise.rs"]
mod elementwise;
#[path = "contract/fixtures.rs"]
mod fixtures;
#[path = "contract/linalg_products.rs"]
mod linalg_products;
#[path = "contract/lu.rs"]
mod lu;
#[path = "contract/reduction.rs"]
mod reduction;
#[path = "contract/reduction_prepared.rs"]
mod reduction_prepared;
#[path = "contract/scan.rs"]
mod scan;
#[path = "contract/sparse.rs"]
mod sparse;
pub(crate) use fixtures::*;
