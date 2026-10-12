mod bound;
mod fixtures;
mod grouped;
mod stream;

#[test]
fn module_cases_share_process_state() {
    crate::test_support::run_cases(&[
        (
            "command_stream_dispatches_prepared_wgsl_kernel",
            stream::command_stream_dispatches_prepared_wgsl_kernel as fn(),
        ),
        (
            "command_stream_preserves_order_for_fill_copy_and_dispatch",
            stream::command_stream_preserves_order_for_fill_copy_and_dispatch as fn(),
        ),
        (
            "command_stream_rejects_binding_contract_mismatch",
            stream::command_stream_rejects_binding_contract_mismatch as fn(),
        ),
        (
            "command_stream_copy_rejects_length_mismatch",
            stream::command_stream_copy_rejects_length_mismatch as fn(),
        ),
        (
            "command_stream_copy_prefix_preserves_the_destination_suffix",
            stream::command_stream_copy_prefix_preserves_the_destination_suffix as fn(),
        ),
        (
            "grouped_command_stream_dispatches_multi_group_wgsl_kernel",
            grouped::grouped_command_stream_dispatches_multi_group_wgsl_kernel as fn(),
        ),
        (
            "grouped_sequence_preserves_order_in_one_wgpu_pass",
            grouped::grouped_sequence_preserves_order_in_one_wgpu_pass as fn(),
        ),
        (
            "bound_grouped_sequence_reuses_fixed_resources",
            bound::bound_grouped_sequence_reuses_fixed_resources as fn(),
        ),
        (
            "bound_grouped_dispatch_updates_retained_parameters",
            bound::bound_grouped_dispatch_updates_retained_parameters as fn(),
        ),
        (
            "bound_grouped_parameter_update_rejects_foreign_device",
            bound::bound_grouped_parameter_update_rejects_foreign_device as fn(),
        ),
        (
            "bound_grouped_dispatch_rejects_invalid_parameter_size",
            bound::bound_grouped_dispatch_rejects_invalid_parameter_size as fn(),
        ),
        (
            "bound_grouped_sequence_rejects_foreign_device",
            bound::bound_grouped_sequence_rejects_foreign_device as fn(),
        ),
        (
            "bound_grouped_dispatch_rejects_group_mismatch",
            bound::bound_grouped_dispatch_rejects_group_mismatch as fn(),
        ),
        (
            "grouped_command_stream_rejects_group_mismatch",
            grouped::grouped_command_stream_rejects_group_mismatch as fn(),
        ),
        (
            "kernel_interface_declares_wgpu_parameter_binding_after_storage",
            grouped::kernel_interface_declares_wgpu_parameter_binding_after_storage as fn(),
        ),
    ]);
}
