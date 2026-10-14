use super::*;
use crate::domain::dialect::{CudaC, DialectScalar, HipC};
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, CumProdOp, CumSumOp};

#[test]
fn c_family_source_declares_the_tiled_shared_memory_contract() {
    let width = BlockWidth::new(8).expect("non-zero test width");
    let source = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(width);
    assert!(source.contains("extern __shared__ float partial[];"));
    assert!(source.contains("unsigned int line = blockIdx.x;"));
    assert!(source.contains("__syncthreads();"));
    assert!(source.contains("unsigned int chunk_len = (len + 8u - 1u) / 8u;"));
    assert!(source.contains("void scan_kernel("));
}

#[test]
fn cuda_and_hip_scan_sources_are_identical() {
    // Both dialects compile the same C++; the shared emitter is why
    // neither backend carries a copy of this template.
    let width = BlockWidth::new(64).expect("non-zero test width");
    assert_eq!(
        <CudaC as AxisScanDialect>::axis_scan_source::<CumSumOp, f32>(width),
        <HipC as AxisScanDialect>::axis_scan_source::<CumSumOp, f32>(width)
    );
}

#[test]
fn reduced_precision_source_carries_the_scalar_prelude() {
    let width = BlockWidth::new(32).expect("non-zero test width");
    let f16 = c_family_axis_scan_source::<CudaC, CumSumOp, eunomia::F16>(width);
    assert!(f16.starts_with(<eunomia::F16 as DialectScalar<CudaC>>::PRELUDE));
    let bf16 = c_family_axis_scan_source::<CudaC, CumSumOp, eunomia::Bf16>(width);
    assert!(bf16.starts_with(<eunomia::Bf16 as DialectScalar<CudaC>>::PRELUDE));
}

#[test]
fn source_is_sensitive_to_operator_and_scalar() {
    let width = BlockWidth::new(32).expect("non-zero test width");
    let sum = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(width);
    let prod = c_family_axis_scan_source::<CudaC, CumProdOp, f32>(width);
    let sum_u32 = c_family_axis_scan_source::<CudaC, CumSumOp, u32>(width);

    assert!(sum.contains(<CumSumOp as CombineExpr<CudaC>>::EXPR));
    assert!(prod.contains(<CumProdOp as CombineExpr<CudaC>>::EXPR));
    assert_ne!(sum, prod);
    assert!(sum.contains("float partial[]"));
    assert!(sum_u32.contains("unsigned int partial[]"));
    assert_ne!(sum, sum_u32);
}

#[test]
fn width_specializes_the_emitted_chunking() {
    let narrow = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(
        BlockWidth::new(16).expect("non-zero test width"),
    );
    let wide = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(
        BlockWidth::new(256).expect("non-zero test width"),
    );
    assert!(narrow.contains("(len + 16u - 1u) / 16u"));
    assert!(wide.contains("(len + 256u - 1u) / 256u"));
}
