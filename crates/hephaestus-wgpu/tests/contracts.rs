//! Consolidated WGPU integration-test harness.
//!
//! Nextest executes each `#[test]` in a separate process. These named cases run
//! as one suite so process-local device caches remain reusable without reducing
//! the value-semantic oracle set.

use hephaestus_wgpu::{HephaestusError, WgpuDevice};

#[path = "device_features.rs"]
mod device_features;

#[path = "metal_acquisition.rs"]
mod metal_acquisition;

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<WgpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let require_metal = std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some();
            let acquired = if require_metal {
                WgpuDevice::try_metal("hephaestus-integration-contracts")
            } else {
                WgpuDevice::try_default("hephaestus-integration-contracts")
            };
            match acquired {
                Ok(device) => {
                    if require_metal {
                        assert_eq!(
                            device
                                .adapter_info()
                                .expect("Metal adapter identity")
                                .backend,
                            hephaestus_wgpu::wgpu::Backend::Metal
                        );
                    }
                    Some(device)
                }
                Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
                    if require_metal || std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some()
                    {
                        panic!("WGPU adapter required, but acquisition failed: {error}");
                    }
                    eprintln!("skipping WGPU integration contracts: adapter unavailable");
                    None
                }
                Err(error) => panic!("WGPU integration contracts require a provider: {error}"),
            }
        })
        .clone()
}

#[path = "accuracy_sweeps.rs"]
mod accuracy_sweeps;

#[path = "adaptive_pooling_contracts.rs"]
mod adaptive_pooling_contracts;

#[path = "arg_reduce_contracts.rs"]
mod arg_reduce_contracts;

#[path = "attention_contracts.rs"]
mod attention_contracts;

#[path = "axis_reduction_contracts.rs"]
mod axis_reduction_contracts;

#[path = "contract.rs"]
mod contract;

#[path = "convolution_contracts.rs"]
mod convolution_contracts;

#[path = "cross_product_contracts.rs"]
mod cross_product_contracts;

#[path = "ctc_contracts.rs"]
mod ctc_contracts;

#[path = "cat_contracts.rs"]
mod cat_contracts;

#[cfg(feature = "decomposition")]
#[path = "decomposition_contracts.rs"]
mod decomposition_contracts;

#[path = "dense_product_contracts.rs"]
mod dense_product_contracts;

#[path = "dense_vector.rs"]
mod dense_vector;

#[path = "dense_vector_contracts.rs"]
mod dense_vector_contracts;

#[path = "dot_contracts.rs"]
mod dot_contracts;

#[path = "elementwise_contracts.rs"]
mod elementwise_contracts;

#[path = "embedding_contracts.rs"]
mod embedding_contracts;

#[path = "full_reduction_contracts.rs"]
mod full_reduction_contracts;

#[path = "fft_contracts.rs"]
mod fft_contracts;

#[path = "fusion_contracts.rs"]
mod fusion_contracts;

#[path = "fixed_fd_contracts.rs"]
mod fixed_fd_contracts;

#[path = "gather_contracts.rs"]
mod gather_contracts;

#[path = "index_put_contracts.rs"]
mod index_put_contracts;

#[path = "index_select_contracts.rs"]
mod index_select_contracts;

#[path = "interpolation_contracts.rs"]
mod interpolation_contracts;

#[path = "masked_fill_contracts.rs"]
mod masked_fill_contracts;

#[path = "nonzero_contracts.rs"]
mod nonzero_contracts;

#[path = "pad_contracts.rs"]
mod pad_contracts;

#[path = "parameterized_unary_contracts.rs"]
mod parameterized_unary_contracts;

#[path = "random_contracts.rs"]
mod random_contracts;

#[path = "ray_integral_contracts.rs"]
mod ray_integral_contracts;

#[path = "retained_reduction_contracts.rs"]
mod retained_reduction_contracts;

#[path = "roll_contracts.rs"]
mod roll_contracts;

#[path = "scan_contracts.rs"]
mod scan_contracts;

#[path = "scatter_contracts.rs"]
mod scatter_contracts;

#[path = "seam_contracts.rs"]
mod seam_contracts;

#[path = "sort_contracts.rs"]
mod sort_contracts;

#[cfg(feature = "sparse")]
#[path = "sparse_contracts.rs"]
mod sparse_contracts;

#[path = "split_contracts.rs"]
mod split_contracts;

#[path = "stack_contracts.rs"]
mod stack_contracts;

#[path = "stateful_update_contracts.rs"]
mod stateful_update_contracts;

#[path = "stencil_contracts.rs"]
mod stencil_contracts;

#[path = "topk_contracts.rs"]
mod topk_contracts;

#[path = "staggered_contracts.rs"]
mod staggered_contracts;

#[path = "staggered3d.rs"]
mod staggered3d;

#[path = "stencil_laplacian.rs"]
mod stencil_laplacian;

#[path = "strided.rs"]
mod strided;

#[path = "tile_contracts.rs"]
mod tile_contracts;

#[path = "transfer_contracts.rs"]
mod transfer_contracts;

#[path = "triangular_contracts.rs"]
mod triangular_contracts;

#[path = "typed_elementwise.rs"]
mod typed_elementwise;

#[path = "typed_elementwise_contracts.rs"]
mod typed_elementwise_contracts;

#[path = "volume_ray_integral.rs"]
mod volume_ray_integral;

#[path = "wgpu_reexport.rs"]
mod wgpu_reexport;

#[path = "where_contracts.rs"]
mod where_contracts;

#[path = "contract_case_registry.rs"]
mod contract_case_registry;

use contract_case_registry::{CONTRACT_CASES, EXPECTED_CONTRACT_CASES};

#[test]
fn integration_contract_cases_share_process_devices() {
    // Cache setup failure before case-level panic aggregation so a required
    // adapter failure is reported once rather than retried by every GPU case.
    let _cached_device = device_or_skip();
    assert_eq!(
        CONTRACT_CASES.len(),
        EXPECTED_CONTRACT_CASES,
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
