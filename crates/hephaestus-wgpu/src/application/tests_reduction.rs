use super::*;

#[test]
fn axis_zero_tile_preserves_the_requested_workgroup_width() {
    assert_eq!(axis0_tile_shape(BlockWidth::DEFAULT), (64, 4));
    assert_eq!(
        axis0_tile_shape(BlockWidth::new(32).expect("non-zero width")),
        (32, 1)
    );
}

#[test]
fn f16_reduction_shaders_carry_the_enable_prelude() {
    use hephaestus_core::SumOp;

    let width = BlockWidth::DEFAULT;
    for source in [
        shader_source::<SumOp, eunomia::F16>(width),
        final_reduction_shader_source::<SumOp, eunomia::F16>(width),
        axis_reduction_parallel_shader_source::<SumOp, eunomia::F16>(width),
        mean_axis_parallel_shader_source::<eunomia::F16>(width),
        axis_reduction_axis0_tiled_shader_source::<SumOp, eunomia::F16>(width),
        mean_axis0_tiled_shader_source::<eunomia::F16>(width),
    ] {
        assert!(
            source.starts_with("enable f16;"),
            "F16 shader must open with the enable directive"
        );
    }
    // Empty prelude for f32: shaders are byte-identical to before.
    assert!(
        shader_source::<SumOp, f32>(width).starts_with("@group"),
        "f32 shader must not gain a prelude"
    );
}
