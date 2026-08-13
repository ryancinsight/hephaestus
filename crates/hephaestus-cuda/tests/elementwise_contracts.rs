//! CUDA instantiation of the shared untyped elementwise clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_elementwise_contract;
use hephaestus_core::{CudaC, EqOp, GeOp, GtOp, LeOp, LtOp, NeOp, TypedBinaryExpr};
use hephaestus_cuda::{CudaDevice, CudaElementwiseOps};

#[test]
fn cuda_satisfies_the_elementwise_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA elementwise conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA elementwise conformance requires a physical device: {error}"),
    };
    assert_elementwise_contract(&device, &CudaElementwiseOps);
}

#[test]
fn cuda_f64_comparison_expressions_are_provider_typed() {
    assert_eq!(
        <EqOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs == rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <NeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs != rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <LtOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs < rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GtOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs > rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <LeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs <= rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs >= rhs ? 1.0 : 0.0"
    );
}
