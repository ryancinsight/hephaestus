#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Value-semantic contracts for the ROCm device substrate.
//!
//! The default build verifies the typed unavailable path. With `--features
//! rocm`, the same tests execute against HIP when an AMD device is present.
//! Hardware CI sets `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` so an unavailable device
//! fails that lane instead of being reported as device evidence.

use hephaestus_core::{
    BinaryStorageKernel, Binding, BindingDecl, BlockWidth, CommandStream, ComputeDevice,
    ComputeDeviceCapabilities, DenseVectorOps, DeviceBuffer, DeviceFeature, DispatchGrid,
    GroupedBinding, GroupedBindingDecl, GroupedCommandStream, GroupedKernelDevice,
    GroupedKernelInterface, GroupedKernelSequence, GroupedKernelSource, HephaestusError, HipC,
    KernelDevice, KernelInterface, KernelSource, MaxOp, MinOp, SumOp,
};
#[cfg(feature = "decomposition")]
use hephaestus_rocm::MatrixDecompose;
use hephaestus_rocm::{
    AbsOp, AddOp, AsGpuMatrixOperand, CosOp, CumSumOp, DivOp, EluGradOp, EluOp, EqOp, ExpNegOp,
    ExpOp, GeluTanhGradOp, GeluTanhOp, GpuCsrMatrix, IdentityOp, LnOp, MatrixFunction, MatrixNorm,
    MatrixProduct, MatrixProperties, MatrixSolve, MishGradOp, MishOp, MulOp, NegOp, PowOp, ProdOp,
    RecipOp, Result, RocmDevice, RocmMultiStorageKernel, RocmVectorOps, ScanDirection, SiluGradOp,
    SiluOp, SinOp, SoftplusGradOp, SoftplusOp, SqrtOp, StridedOperand, SubOp, batched_matmul,
    batched_matmul_into, binary_elementwise, binary_elementwise_into, binary_elementwise_strided,
    binary_elementwise_strided_into, binary_elementwise_strided_typed, cumprod, cumprod_into,
    cumsum, dot, kron, kron_into, matmul, matmul_into, matpow, max_axis, mean_axis, mean_axis_into,
    min_axis, norm_l1, norm_l2, norm_max, normal_with_seed, prepare_dot, prepare_max_axis_into,
    prepare_mean_axis_into, prepare_norm_l2, prepare_reduction, prepare_reduction_with_width,
    prepare_sum_axis_into, prod_axis, prod_axis_into, reduce_axis, reduction_with_width,
    scalar_elementwise, scalar_elementwise_strided, scalar_elementwise_strided_into, scan_axis,
    scan_axis_into, spmm, spmm_into, spmv, spmv_many, spmv_many_into,
    submit_prepared_axis_reduction_batch, submit_prepared_reduction_batch, suffix_prod,
    suffix_prod_into, suffix_sum, suffix_sum_into, sum_axis, trace, unary_elementwise,
    unary_elementwise_strided, unary_elementwise_strided_into, uniform_with_seed,
};
#[cfg(feature = "decomposition")]
use hephaestus_rocm::{
    bidiagonalize, bunch_kaufman, cholesky_decompose, cholesky_decompose_blocked, col_piv_qr,
    col_piv_qr_blocked, eigenvalues, full_piv_lu, full_piv_lu_blocked, hessenberg, lu_decompose,
    lu_decompose_blocked, qr_decompose, qr_decompose_blocked, schur, singular_values,
    svd_decompose, svd_rank_revealing, symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi,
    udu_decompose,
};
use leto::Layout;

impl hephaestus_core::DialectScalar<HipC> for AlternateIdentity {
    const TYPE_TOKEN: &'static str = "int";
}

impl hephaestus_rocm::MatrixIdentityScalar for AlternateIdentity {
    const ZERO: Self = Self(-7);
    const ONE: Self = Self(9);
}
use std::borrow::Cow;

impl KernelInterface for StreamAddKernel {
    type Params = u32;

    const LABEL: &'static str = "rocm-contract-stream-add";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl KernelSource<HipC> for StreamAddKernel {
    const ENTRY: &'static str = "rocm_contract_stream_add";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
extern "C" __global__ void rocm_contract_stream_add(
    const float* lhs,
    const float* rhs,
    float* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#,
        )
    }
}

impl GroupedKernelInterface for GroupedStreamAddKernel {
    type Params = u32;

    const LABEL: &'static str = "rocm-contract-grouped-stream-add";
    const BINDINGS: &'static [GroupedBindingDecl] = &[
        GroupedBindingDecl::read_only::<f32>(0, 0),
        GroupedBindingDecl::read_only::<f32>(0, 1),
        GroupedBindingDecl::read_write::<f32>(0, 2),
    ];
    const PARAM_GROUP: u32 = 0;
    const PARAM_BINDING: u32 = 3;
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl GroupedKernelSource<HipC> for GroupedStreamAddKernel {
    const ENTRY: &'static str = "rocm_contract_grouped_stream_add";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
extern "C" __global__ void rocm_contract_grouped_stream_add(
    const float* lhs,
    const float* rhs,
    float* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#,
        )
    }
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
