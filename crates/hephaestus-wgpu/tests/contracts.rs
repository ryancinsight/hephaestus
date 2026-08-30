//! Consolidated WGPU integration-test harness.
//!
//! Nextest executes each `#[test]` in a separate process. These named cases run
//! as one suite so process-local device caches remain reusable without reducing
//! the value-semantic oracle set.

use hephaestus_wgpu::{HephaestusError, WgpuDevice};

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<WgpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(
            || match WgpuDevice::try_default("hephaestus-integration-contracts") {
                Ok(device) => Some(device),
                Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
                    if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                        panic!("WGPU adapter required, but acquisition failed: {error}");
                    }
                    eprintln!("skipping WGPU integration contracts: adapter unavailable");
                    None
                }
                Err(error) => panic!("WGPU integration contracts require a provider: {error}"),
            },
        )
        .clone()
}

#[path = "attention_contracts.rs"]
mod attention_contracts;

#[path = "axis_reduction_contracts.rs"]
mod axis_reduction_contracts;

#[path = "contract.rs"]
mod contract;

#[path = "convolution_contracts.rs"]
mod convolution_contracts;

#[path = "decomposition_contracts.rs"]
mod decomposition_contracts;

#[path = "dense_product_contracts.rs"]
mod dense_product_contracts;

#[path = "dense_vector.rs"]
mod dense_vector;

#[path = "dense_vector_contracts.rs"]
mod dense_vector_contracts;

#[path = "elementwise_contracts.rs"]
mod elementwise_contracts;

#[path = "full_reduction_contracts.rs"]
mod full_reduction_contracts;

#[path = "fft_contracts.rs"]
mod fft_contracts;

#[path = "parameterized_unary_contracts.rs"]
mod parameterized_unary_contracts;

#[path = "random_contracts.rs"]
mod random_contracts;

#[path = "ray_integral_contracts.rs"]
mod ray_integral_contracts;

#[path = "scan_contracts.rs"]
mod scan_contracts;

#[path = "seam_contracts.rs"]
mod seam_contracts;

#[path = "sparse_contracts.rs"]
mod sparse_contracts;

#[path = "stateful_update_contracts.rs"]
mod stateful_update_contracts;

#[path = "stencil_contracts.rs"]
mod stencil_contracts;

#[path = "stencil_laplacian.rs"]
mod stencil_laplacian;

#[path = "strided.rs"]
mod strided;

#[path = "transfer_contracts.rs"]
mod transfer_contracts;

#[path = "typed_elementwise.rs"]
mod typed_elementwise;

#[path = "typed_elementwise_contracts.rs"]
mod typed_elementwise_contracts;

#[path = "volume_ray_integral.rs"]
mod volume_ray_integral;

#[path = "wgpu_reexport.rs"]
mod wgpu_reexport;

type ContractCase = (&'static str, fn());

macro_rules! contract_cases {
    ($($(#[$meta:meta])* $case:path),+ $(,)?) => {
        const CONTRACT_CASES: &[ContractCase] = &[
            $(
                $(#[$meta])*
                (stringify!($case), $case)
            ),+
        ];
    };
}

contract_cases!(
    contract::upload_download_round_trips_values,
    contract::device_local_copy_preserves_values_and_rejects_mismatch,
    contract::uninitialized_allocation_is_fully_overwritten_before_read,
    contract::odd_u16_storage_preserves_logical_values_when_device_exists,
    contract::test_placement_aware_allocation,
    contract::download_rejects_length_mismatch,
    contract::elementwise_add_matches_cpu_reference,
    contract::elementwise_mul_matches_cpu_reference_integral,
    contract::elementwise_rejects_input_length_mismatch,
    contract::elementwise_into_reuses_caller_output_buffers,
    contract::elementwise_into_rejects_output_input_aliasing,
    contract::elementwise_unary_matches_cpu_reference,
    contract::elementwise_lgamma_matches_cpu_reference,
    contract::elementwise_activation_markers_match_cpu_reference,
    contract::elementwise_scalar_matches_cpu_reference,
    contract::reduction_sum_matches_cpu_reference,
    contract::reduction_min_max_matches_cpu_reference,
    contract::reduction_width_is_part_of_dispatch_contract,
    contract::axis_reductions_match_leto_reference,
    contract::mixed_prepared_reduction_batch_preserves_scalar_and_axis_results,
    contract::axis_reduction_grid_stride_matches_leto_reference_beyond_block_width,
    contract::axis_zero_tiling_matches_exact_multi_tile_oracles,
    contract::axis_scans_match_leto_reference,
    contract::cumprod_convenience_preserves_strided_and_empty_contract,
    contract::axis_scan_long_line_matches_leto_reference,
    contract::acquisition_reports_themis_topology_from_adapter,
    contract::linalg_matmul_matches_cpu_reference,
    contract::linalg_batched_matmul_matches_cpu_reference,
    contract::linalg_kron_matches_leto_reference,
    contract::linalg_matpow_matches_leto_reference,
    contract::linalg_matpow_rejects_non_square,
    contract::linalg_dot_matches_cpu_reference,
    contract::prepared_dot_reuses_output_and_observes_input_updates,
    contract::linalg_trace_matches_cpu_reference,
    contract::linalg_matrix_rank_matches_leto_reference,
    contract::linalg_det_matches_leto_reference,
    contract::matrix_rank_relative_tolerance_is_the_discriminator,
    contract::det_of_near_singular_triangular_is_exact_pivot_product,
    contract::blocked_cholesky_matches_leto_reference_across_block_boundary,
    contract::blocked_cholesky_zeroes_strict_upper_outside_diagonal_blocks,
    contract::symmetric_eigen_jacobi_rejects_non_symmetric_input,
    contract::eigenvalues_match_closed_form_diagonal,
    contract::eigenvalues_match_exact_complex_pair_blocks,
    contract::eigenvalues_match_structured_and_dense_leto_oracles,
    contract::eigenvalues_symmetric_input_is_real_and_matches_leto,
    contract::eigenvalues_rejects_non_square_input,
    contract::singular_values_match_closed_form_diagonal,
    contract::svd_decompose_reconstructs_leto_reference,
    contract::svd_rank_revealing_accepts_rank_deficient_matrix,
    contract::bidiagonalize_reconstructs_and_preserves_singular_values,
    contract::bidiagonalize_rejects_wide_matrix,
    contract::schur_reconstructs_quasi_triangular_and_preserves_spectrum,
    contract::schur_rejects_rectangular_matrix,
    contract::hessenberg_reconstructs_and_preserves_similarity_invariants,
    contract::hessenberg_rejects_rectangular_matrix,
    contract::full_piv_lu_reconstructs_and_matches_leto_oracles,
    contract::full_piv_lu_reveals_rank_deficiency_and_rejects_inverse,
    contract::full_piv_lu_rejects_rectangular_matrix,
    contract::blocked_pivoted_decompositions_match_ordinary_contracts,
    contract::write_buffer_overwrites_existing_data,
    contract::write_buffer_rejects_length_mismatch,
    contract::write_buffer_empty_is_noop,
    contract::write_buffer_integer_types,
    contract::write_sub_buffer_overwrites_only_requested_range,
    contract::write_sub_buffer_rejects_out_of_range_write,
    contract::write_sub_buffer_empty_tail_write_is_noop,
    contract::cholesky_rejects_singular_matrix,
    contract::lu_rejects_singular_matrix,
    contract::linalg_norms_match_cpu_reference,
    contract::prepared_l2_norm_reuses_output_and_observes_input_updates,
    contract::linalg_reductions_accept_strided_views,
    contract::blocked_lu_matches_leto_reference,
    contract::blocked_lu_identity_yields_identity_factors,
    contract::blocked_lu_solve_known_system_accurate,
    contract::blocked_lu_rejects_singular_matrix,
    contract::blocked_qr_matches_leto_reference,
    contract::qr_r_buffer_is_upper_triangular_on_both_entry_points,
    contract::blocked_qr_preserves_panel_boundary_contracts,
    contract::blocked_qr_identity_yields_identity_r,
    contract::blocked_qr_solve_known_system_accurate,
    contract::blocked_qr_rejects_underdetermined,
    contract::blocked_cholesky_identity_yields_identity_lower,
    contract::blocked_cholesky_spd_reconstruction_matches_original,
    contract::blocked_cholesky_solve_known_system_accurate,
    contract::blocked_cholesky_rejects_singular_matrix,
    contract::udu_decompose_rejects_invalid_contracts,
    contract::bunch_kaufman_rejects_rectangular_and_nonsymmetric,
    contract::linalg_pinv_matches_closed_form_diagonal,
    contract::linalg_pinv_rank_deficient_satisfies_moore_penrose,
    contract::linalg_pinv_handles_rectangular_full_rank_matrix,
    contract::linalg_pinv_rejects_non_finite_input,
    contract::linalg_matexp_matches_closed_form_diagonal,
    contract::linalg_matexp_matches_nilpotent_and_rotation_closed_forms,
    contract::linalg_matexp_matches_leto_general_matrix,
    contract::linalg_matexp_rejects_invalid_contracts,
    contract::test_wgpu_uniform_and_normal_with_seed,
    contract::test_wgpu_sparse_matrix_spmv_spmm,
    contract::blocked_cholesky_rejects_non_dense_operands,
    contract::blocked_lu_rejects_non_dense_operands,
    contract::blocked_qr_rejects_non_dense_operands,
    contract::blocked_pivoted_decompositions_reject_non_dense_operands,
    contract::empty_qr_preserves_shape_and_identity,
    contract::fdtd_3d_provider_matches_sequential_cpu_reference,
    attention_contracts::wgpu_satisfies_the_attention_contract,
    attention_contracts::prepared_dispatch_resets_semantic_status_after_failure,
    attention_contracts::zero_probability_prefix_preserves_stable_convex_output,
    axis_reduction_contracts::wgpu_satisfies_the_axis_reduction_contract,
    convolution_contracts::wgpu_satisfies_the_convolution_contract,
    decomposition_contracts::non_blocked_decomposition_heap_readbacks_are_provider_owned,
    decomposition_contracts::wgpu_satisfies_the_decomposition_contract,
    dense_product_contracts::wgpu_satisfies_the_dense_product_contract,
    dense_vector_contracts::wgpu_satisfies_the_dense_vector_contract,
    dense_vector::axpy_matches_the_cpu_reference,
    dense_vector::xpay_matches_the_cpu_reference,
    dense_vector::scale_and_copy_match_the_cpu_reference,
    dense_vector::subtract_matches_the_cpu_reference,
    dense_vector::reductions_match_the_cpu_reference,
    dense_vector::prepared_reductions_reuse_their_bound_allocation,
    dense_vector::mismatched_lengths_are_rejected,
    elementwise_contracts::wgpu_satisfies_the_elementwise_contract,
    fft_contracts::prepared_fft_device_preflight_is_public_and_typed,
    full_reduction_contracts::wgpu_satisfies_the_full_reduction_contract,
    parameterized_unary_contracts::wgpu_satisfies_the_parameterized_unary_contract,
    #[cfg(any(feature = "decomposition", feature = "sparse"))]
    random_contracts::wgpu_satisfies_the_random_init_contract,
    ray_integral_contracts::wgpu_satisfies_the_ray_integral_contract,
    scan_contracts::wgpu_satisfies_the_scan_contract,
    seam_contracts::full_reduction_honors_output_offset,
    seam_contracts::empty_full_reductions_write_operator_identities,
    seam_contracts::prepared_elementwise_rejects_cross_device_dispatch,
    seam_contracts::prepared_scan_and_full_reduction_reject_cross_device_dispatch,
    seam_contracts::overlapping_writable_layouts_fail_before_mutation,
    seam_contracts::elementwise_and_scan_match_value_oracles,
    seam_contracts::invalid_external_expression_is_a_typed_preparation_error,
    seam_contracts::invalid_external_combine_is_a_typed_preparation_error,
    seam_contracts::full_reduction_rejects_foreign_buffers_before_mutation,
    sparse_contracts::wgpu_satisfies_the_sparse_operator_contract,
    stateful_update_contracts::wgpu_satisfies_the_stateful_update_contract,
    stateful_update_contracts::foreign_device_buffers_fail_before_mutation,
    stencil_contracts::wgpu_satisfies_the_stencil_contract,
    stencil_laplacian::laplacian_minimum_grid_matches_cpu_reference,
    stencil_laplacian::laplacian_dirichlet_matches_cpu_reference,
    stencil_laplacian::laplacian_neumann_matches_cpu_reference,
    stencil_laplacian::laplacian_periodic_matches_cpu_reference,
    stencil_laplacian::laplacian_non_square_2x3_matches_cpu_reference,
    stencil_laplacian::laplacian_non_square_3x2_matches_cpu_reference,
    stencil_laplacian::laplacian_large_dirichlet_16x16_matches_cpu_reference,
    stencil_laplacian::laplacian_large_periodic_16x16_matches_cpu_reference,
    stencil_laplacian::laplacian_storage_length_mismatch_is_rejected_before_launch,
    strided::strided_add_transposed_input_matches_cpu,
    strided::strided_broadcast_inputs_match_cpu,
    strided::strided_offset_output_writes_only_selected_region,
    strided::strided_rejects_aliasing_output_and_short_buffers,
    strided::strided_rank3_batched_matches_cpu,
    strided::strided_unary_transposed_matches_cpu,
    strided::strided_unary_broadcasts_input_to_output_shape,
    strided::strided_scalar_matches_binary_broadcast_semantics,
    strided::non_default_block_width_produces_identical_results,
    transfer_contracts::wgpu_satisfies_the_transfer_contract,
    typed_elementwise_contracts::wgpu_satisfies_the_typed_elementwise_contract,
    typed_elementwise::typed_comparisons_are_exact_indicators_for_unsigned_operands,
    typed_elementwise::typed_comparisons_order_signed_operands_by_sign,
    typed_elementwise::typed_comparisons_are_exact_indicators_for_finite_floats,
    typed_elementwise::typed_comparison_into_writes_caller_storage_and_matches_allocating_form,
    typed_elementwise::typed_comparison_rejects_length_mismatch,
    typed_elementwise::typed_comparison_strided_into_respects_source_strides,
    typed_elementwise::typed_comparison_strided_broadcasts_into_dense_output,
    volume_ray_integral::uniform_field_integrates_to_value_times_chord,
    volume_ray_integral::affine_field_is_integrated_exactly_by_midpoint,
    volume_ray_integral::step_size_does_not_change_a_uniform_integral,
    volume_ray_integral::oblique_ray_matches_cpu_reference,
    wgpu_reexport::provider_exports_wgpu_abi_types,
);

#[test]
fn integration_contract_cases_share_process_devices() {
    // Cache setup failure before case-level panic aggregation so a required
    // adapter failure is reported once rather than retried by every GPU case.
    let _cached_device = device_or_skip();
    assert_eq!(
        CONTRACT_CASES.len(),
        171,
        "the consolidated integration contract must retain every migrated case"
    );

    let failures = CONTRACT_CASES
        .iter()
        .filter_map(|(name, case)| std::panic::catch_unwind(case).is_err().then_some(*name))
        .collect::<Vec<_>>();

    assert!(
        failures.is_empty(),
        "integration contract cases failed: {}",
        failures.join(", ")
    );
}
