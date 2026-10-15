use super::*;

use hephaestus_core::{BlockWidth, CudaC, DialectScalar, SumOp};
use leto::Layout;

#[test]
fn reduced_precision_sources_carry_the_scalar_prelude() {
    let width = BlockWidth::DEFAULT;
    let f16_prelude = <eunomia::F16 as DialectScalar<CudaC>>::PRELUDE;
    let bf16_prelude = <eunomia::Bf16 as DialectScalar<CudaC>>::PRELUDE;
    assert!(shader_source::<SumOp, eunomia::F16>(width).starts_with(f16_prelude));
    assert!(shader_source::<SumOp, eunomia::Bf16>(width).starts_with(bf16_prelude));
    assert!(axis_reduction_shader_source::<SumOp, eunomia::F16>().starts_with(f16_prelude));
    assert!(axis_reduction_shader_source::<SumOp, eunomia::Bf16>().starts_with(bf16_prelude));
    assert!(mean_axis_shader_source::<eunomia::F16>().starts_with(f16_prelude));
    assert!(mean_axis_shader_source::<eunomia::Bf16>().starts_with(bf16_prelude));
}

#[test]
fn dense_scalar_layout_accepts_canonical_spellings() {
    let row = Layout::c_contiguous([1, 6]).expect("valid test layout");
    assert!(is_dense_scalar_layout(&row, 1));
    assert!(!is_dense_scalar_layout(&row, 0));
    let col = Layout::c_contiguous([6, 1]).expect("valid test layout");
    assert!(is_dense_scalar_layout(&col, 0));
    assert!(!is_dense_scalar_layout(&col, 1));
}

#[test]
fn dense_scalar_layout_ignores_unit_axis_strides() {
    // Left-padded consumer spelling (Coeus `ranked::<2>` over a rank-1
    // input): strides `[0, 1]` address the same dense block as `[6, 1]`.
    let padded = Layout::try_new([1, 6], [0, 1], 0).expect("valid test layout");
    assert!(is_dense_scalar_layout(&padded, 1));
    assert!(!is_dense_scalar_layout(&padded, 0));
    let padded_col = Layout::try_new([6, 1], [1, 0], 0).expect("valid test layout");
    assert!(is_dense_scalar_layout(&padded_col, 0));
    assert!(!is_dense_scalar_layout(&padded_col, 1));
}

#[test]
fn dense_scalar_layout_rejects_strided_offset_and_wide() {
    let strided = Layout::try_new([1, 6], [0, 2], 0).expect("valid test layout");
    assert!(!is_dense_scalar_layout(&strided, 1));
    let offset = Layout::try_new([1, 6], [6, 1], 4).expect("valid test layout");
    assert!(!is_dense_scalar_layout(&offset, 1));
    let wide = Layout::c_contiguous([2, 6]).expect("valid test layout");
    assert!(!is_dense_scalar_layout(&wide, 1));
    assert!(!is_dense_scalar_layout(&wide, 0));
}
