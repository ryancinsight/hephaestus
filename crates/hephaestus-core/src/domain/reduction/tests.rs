use super::*;

fn width() -> BlockWidth {
    BlockWidth::new(256).expect("non-zero width")
}

#[test]
fn plans_a_dense_axis1_reduction() {
    let input = Layout::c_contiguous([3, 4]).expect("contiguous");
    let output = Layout::c_contiguous([3, 1]).expect("contiguous");
    let plan = plan_axis_reduction(&input, 12, &output, 3, 1, width(), false)
        .expect("valid")
        .expect("non-empty");
    assert_eq!(plan.meta.offsets[2], 1); // axis
    assert_eq!(plan.meta.offsets[3], 3); // output_len
    assert_eq!(plan.groups, 1);
    assert_eq!(plan.meta.input_shape, [3, 4]);
}

#[test]
fn output_shape_must_collapse_the_axis() {
    let input = Layout::c_contiguous([3, 4]).expect("contiguous");
    let wrong = Layout::c_contiguous([3, 4]).expect("contiguous");
    let err = plan_axis_reduction(&input, 12, &wrong, 12, 1, width(), false).unwrap_err();
    assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
}

#[test]
fn aliasing_is_rejected() {
    let input = Layout::c_contiguous([2, 2]).expect("contiguous");
    let output = Layout::c_contiguous([1, 2]).expect("contiguous");
    let err = plan_axis_reduction(&input, 4, &output, 2, 0, width(), true).unwrap_err();
    assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
}

#[test]
fn empty_output_is_a_noop() {
    let input = Layout::c_contiguous([0, 4]).expect("contiguous");
    let output = Layout::c_contiguous([0, 1]).expect("contiguous");
    let plan = plan_axis_reduction(&input, 0, &output, 0, 1, width(), false).expect("valid");
    assert!(plan.is_none());
}

#[test]
fn scalar_pass_count_matches_tree_depth() {
    let width = BlockWidth::new(256).expect("invariant: test width is non-zero");
    assert_eq!(reduction_pass_count(0, width), 0);
    assert_eq!(reduction_pass_count(1, width), 0);
    assert_eq!(reduction_pass_count(2, width), 1);
    assert_eq!(reduction_pass_count(256, width), 1);
    assert_eq!(reduction_pass_count(257, width), 2);
    assert_eq!(reduction_pass_count(65_536, width), 2);

    let narrow = BlockWidth::new(128).expect("invariant: test width is non-zero");
    assert_eq!(reduction_pass_count(16_385, narrow), 3);
}

#[test]
fn scalar_width_must_be_power_of_two() {
    let valid = BlockWidth::new(128).expect("non-zero width");
    assert!(validate_reduction_width(valid).is_ok());

    let invalid = BlockWidth::new(192).expect("non-zero width");
    let err = validate_reduction_width(invalid).unwrap_err();
    assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
}
