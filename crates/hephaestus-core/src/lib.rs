#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! # hephaestus-core
//!
//! GPU-dependency-free contracts for the Atlas shared accelerator substrate
//! (atlas ADR 0001). This crate defines *what a compute device is* — device
//! acquisition results, typed device buffers, and the dispatch seam — without
//! depending on any GPU API. Backend crates (`hephaestus-wgpu`, and a CUDA
//! backend composing `cuda-oxide` + `cutile`) implement these contracts.
//!
//! Consumers (`apollo` GPU transforms, `coeus` GPU tensor backends) program
//! against this seam so spectral and tensor packages share one device layer
//! without an `apollo`→`coeus` dependency edge. Autodiff stays in `coeus`;
//! kernels dispatched here are autodiff-agnostic functions.

/// Device and buffer contracts.
pub mod domain;

pub use domain::buffer::DeviceBuffer;
pub use domain::decomposition::{panel_lu_packed, panel_qr_packed, split_packed_lu};
pub use domain::device::{
    validate_buffer_size, validate_slice_alignment, ComputeDevice, ComputeDeviceAcquisition,
    ComputeDeviceCapabilities, DeviceFeature, DeviceLimits, DevicePreference,
};
pub use domain::dialect::{CudaC, DialectScalar, KernelDialect, Wgsl};
pub use domain::error::{HephaestusError, Result};
pub use domain::interface::{
    Access, BindingDecl, GroupedBindingDecl, GroupedKernelInterface, GroupedKernelSource,
    KernelInterface, KernelSource,
};
pub use domain::kernel::{
    BinaryStorageKernel, DispatchGrid, MultiStorageDevice, MultiStorageKernel, UnaryStorageKernel,
};
pub use domain::launch::BlockWidth;
pub use domain::ops::{
    AbsOp, AddOp, BinaryExpr, CombineExpr, CosOp, CumProdOp, CumSumOp, DivOp, ExpNegOp, ExpOp,
    IdentityOp, IdentityToken, LnOp, MaxOp, MinOp, MulOp, NegOp, OpIdentity, PowOp, RecipOp, SinOp,
    SqrtOp, SubOp, SumOp, UnaryExpr,
};
pub use domain::scan::{plan_axis_scan, AxisScanDispatch, AxisScanMeta, ScanDirection};
pub use domain::stream::{
    validate_bindings, validate_grouped_bindings, Binding, CommandStream, GroupedBinding,
    GroupedCommandStream, GroupedKernelDevice, GroupedKernelSequence, KernelDevice,
};
