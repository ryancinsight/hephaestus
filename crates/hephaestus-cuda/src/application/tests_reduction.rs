use super::*;

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
