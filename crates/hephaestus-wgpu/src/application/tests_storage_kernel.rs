use super::*;

use hephaestus_core::HephaestusError;

#[test]
fn module_cases_share_process_state() {
    crate::test_support::run_cases(&[
        (
            "multi_storage_layout_rejects_parameter_collision",
            multi_storage_layout_rejects_parameter_collision as fn(),
        ),
        (
            "multi_storage_layout_rejects_duplicate_storage_binding",
            multi_storage_layout_rejects_duplicate_storage_binding as fn(),
        ),
    ]);
}

fn multi_storage_layout_rejects_parameter_collision() {
    let err = validate_distinct_bindings(
        &[
            WgslStorageBindingLayout::read_only(0),
            WgslStorageBindingLayout::read_write(2),
        ],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
}

fn multi_storage_layout_rejects_duplicate_storage_binding() {
    let err = validate_distinct_bindings(
        &[
            WgslStorageBindingLayout::read_only(0),
            WgslStorageBindingLayout::read_write(0),
        ],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
}
