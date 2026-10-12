#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Contract tests for the CUDA `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without the `cuda` feature or without a CUDA device,
//! [`CudaDevice::try_default`] returns `Err` and each test skips.
//! Hardware CI sets `HEPHAESTUS_CUDA_REQUIRE_DEVICE=1` so acquisition failure
//! fails the lane instead of being reported as device evidence.

use hephaestus_core::test_support::assert_rejects;
use hephaestus_core::{
    BlockWidth, ComputeDevice, ComputeDeviceCapabilities, DenseVectorOps, DeviceBuffer,
    DeviceFeature, HephaestusError, Result,
};
use hephaestus_cuda::{
    AbsOp, AddOp, CudaDevice, CudaVectorOps, CumSumOp, EluGradOp, EluOp, ExpOp, GeluTanhGradOp,
    GeluTanhOp, MaxOp, MinOp, MishGradOp, MishOp, MulOp, NegOp, ProdOp, RecipOp, SiluGradOp,
    SiluOp, SoftplusGradOp, SoftplusOp, SqrtOp, StridedOperand, SubOp, SumOp, batched_matmul,
    batched_matmul_into, binary_elementwise, binary_elementwise_into, cumprod, cumprod_into, dot,
    kron, matmul, matmul_into, matpow, norm_l1, norm_l2, norm_max, prepare_dot,
    prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into, prepare_norm_l2,
    prepare_reduction, prepare_reduction_with_width, prepare_sum_axis_into, prod_axis,
    prod_axis_into, reduce_axis, reduction, reduction_with_width, scalar_elementwise,
    scalar_elementwise_into, scan_axis, submit_prepared_axis_reduction_batch,
    submit_prepared_reduction_batch, suffix_prod, suffix_prod_into, suffix_sum, suffix_sum_into,
    trace, unary_elementwise, unary_elementwise_into,
};
use leto::Layout;

#[path = "contracts/fusion_contracts.rs"]
mod fusion_contracts;

impl hephaestus_core::DialectScalar<hephaestus_core::CudaC> for AlternateIdentity {
    const TYPE_TOKEN: &'static str = "int";
}

impl hephaestus_cuda::MatrixIdentityScalar for AlternateIdentity {
    const ZERO: Self = Self(-7);
    const ONE: Self = Self(9);
}

#[path = "contract/axis.rs"]
mod axis;
#[path = "contract/buffer.rs"]
mod buffer;
#[path = "contract/cholesky.rs"]
mod cholesky;
#[path = "contract/device.rs"]
mod device;
#[path = "contract/eigenvalues.rs"]
mod eigenvalues;
#[path = "contract/elementwise.rs"]
mod elementwise;
#[path = "contract/fixtures.rs"]
mod fixtures;
#[path = "contract/linalg_norms.rs"]
mod linalg_norms;
#[path = "contract/linalg_products.rs"]
mod linalg_products;
#[path = "contract/lu.rs"]
mod lu;
#[path = "contract/qr.rs"]
mod qr;
#[path = "contract/reduction.rs"]
mod reduction;
#[path = "contract/reduction_map.rs"]
mod reduction_map;
#[path = "contract/reduction_prepared.rs"]
mod reduction_prepared;
#[path = "contract/scan.rs"]
mod scan;
#[path = "contract/sparse.rs"]
mod sparse;
#[path = "contract/svd.rs"]
mod svd;
pub(crate) use fixtures::*;

// ── write_buffer tests ────────────────────────────────────────────────

// ── Extended differential decomposition tests ─────────────────────────────

// ── Blocked decomposition differential tests ────────────────────────────

// ────────────────────────────────────────────────────────────────────────
// Telemetry and Decomposition Contract Helper Functions
// ────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────
// Decomposition Contract Tests
// ────────────────────────────────────────────────────────────────────────
