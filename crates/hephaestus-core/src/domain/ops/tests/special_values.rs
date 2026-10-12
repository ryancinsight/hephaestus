use super::*;

#[test]
fn value_oracles_hold_for_every_shipped_real_scalar() {
    assert_value_oracles_hold::<f32>();
    assert_value_oracles_hold::<f64>();
}

#[test]
fn infinite_inputs_resolve_to_the_analytic_limit_for_every_shipped_real_scalar() {
    assert_infinite_inputs_resolve_to_the_analytic_limit::<f32>();
    assert_infinite_inputs_resolve_to_the_analytic_limit::<f64>();
}

#[test]
fn nan_propagates_through_every_comparison_based_value_function_for_every_shipped_real_scalar() {
    assert_nan_propagates_through_every_comparison_based_value_function::<f32>();
    assert_nan_propagates_through_every_comparison_based_value_function::<f64>();
}
