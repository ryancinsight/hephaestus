#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Differential contract tests: wgpu dispatch vs CPU reference.
//!
//! Tests acquire a real adapter; on hosts without one (headless CI without
//! GPU/lavapipe) they skip with a message rather than fabricate a pass.
//! Hosted software-adapter CI sets `HEPHAESTUS_WGPU_REQUIRE_DEVICE=1` so an
//! unavailable adapter fails that lane instead of being reported as evidence.

#[cfg(feature = "decomposition")]
use hephaestus_core::test_support::assert_rejects;
#[path = "contract/allocation.rs"]
mod allocation;
#[path = "contract/recycling.rs"]
mod recycling;
#[path = "contract/storage.rs"]
mod storage;
#[path = "contract/transfer.rs"]
mod transfer;

use eunomia::Zeroable;
use hephaestus_core::{BlockWidth, Fdtd3dOps, Fdtd3dParams, FdtdMedium, FdtdVelocity};
use hephaestus_wgpu::{
    AbsOp, AddOp, ComputeDevice, DeviceBuffer, EluGradOp, EluOp, ExpNegOp, ExpOp, GeluTanhGradOp,
    GeluTanhOp, HephaestusError, LgammaOp, MaxOp, MinOp, MishGradOp, MishOp, MulOp, NegOp, ProdOp,
    RecipOp, SiluGradOp, SiluOp, SoftplusGradOp, SoftplusOp, SqrtOp, SubOp, SumOp, WgpuDevice,
    WgpuFdtd3dOps, binary_elementwise, binary_elementwise_into, cumsum_into, matrix_rank,
    matrix_rank_with_tolerance, max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis,
    min_axis_into, prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into,
    prepare_reduction, prepare_sum_axis_into, prod_axis, prod_axis_into, reduce_axis, reduction,
    reduction_with_width, scalar_elementwise, scalar_elementwise_into,
    submit_prepared_axis_reduction_batch, submit_prepared_mixed_reduction_batch,
    submit_prepared_reduction_batch, sum_axis, sum_axis_into, unary_elementwise,
    unary_elementwise_into,
};

impl hephaestus_core::DialectScalar<hephaestus_core::Wgsl> for AlternateIdentity {
    const TYPE_TOKEN: &'static str = "i32";
}

impl hephaestus_wgpu::MatmulZero for AlternateIdentity {
    const WGSL_ZERO: &'static str = "0";
}

impl hephaestus_wgpu::MatrixIdentityScalar for AlternateIdentity {
    const ZERO: Self = Self(-7);
    const ONE: Self = Self(9);
}

#[path = "contract/axis.rs"]
mod axis;
#[path = "contract/axis_tiling.rs"]
mod axis_tiling;
#[path = "contract/buffer.rs"]
mod buffer;
#[path = "contract/cholesky.rs"]
#[cfg(feature = "decomposition")]
mod cholesky;
#[path = "contract/device.rs"]
mod device;
#[path = "contract/eigenvalues.rs"]
#[cfg(feature = "decomposition")]
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
#[cfg(feature = "decomposition")]
mod lu;
#[path = "contract/qr.rs"]
#[cfg(feature = "decomposition")]
mod qr;
#[path = "contract/reduction.rs"]
mod reduction;
#[path = "contract/reduction_prepared.rs"]
mod reduction_prepared;
#[path = "contract/scan.rs"]
mod scan;
#[path = "contract/sparse.rs"]
#[cfg(feature = "sparse")]
mod sparse;
#[path = "contract/svd.rs"]
mod svd;
pub(crate) use fixtures::*;

pub(crate) use axis::*;
pub(crate) use axis_tiling::*;
pub(crate) use buffer::*;
#[cfg(feature = "decomposition")]
pub(crate) use cholesky::*;
pub(crate) use device::*;
#[cfg(feature = "decomposition")]
pub(crate) use eigenvalues::*;
pub(crate) use elementwise::*;
pub(crate) use linalg_norms::*;
pub(crate) use linalg_products::*;
#[cfg(feature = "decomposition")]
pub(crate) use lu::*;
#[cfg(feature = "decomposition")]
pub(crate) use qr::*;
pub(crate) use reduction::*;
pub(crate) use reduction_prepared::*;
pub(crate) use scan::*;
#[cfg(feature = "sparse")]
pub(crate) use sparse::*;
pub(crate) use svd::*;

// ── write_buffer tests ────────────────────────────────────────────────

// ── Extended differential decomposition tests ─────────────────────────────

// ── Blocked decomposition differential tests ────────────────────────────
