//! CUDA instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

#![cfg(all(feature = "cuda", feature = "decomposition"))]
use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_cuda::{CudaDecompositionOps, CudaDevice};

const DECOMPOSITION_SOURCES: &[(&str, &str)] = &[
    (
        "bidiagonal",
        include_str!("../src/application/decomposition/bidiagonal.rs"),
    ),
    (
        "bunch_kaufman",
        include_str!("../src/application/decomposition/bunch_kaufman.rs"),
    ),
    (
        "cholesky",
        include_str!("../src/application/decomposition/cholesky.rs"),
    ),
    (
        "col_piv_qr",
        include_str!("../src/application/decomposition/col_piv_qr.rs"),
    ),
    (
        "eigen",
        include_str!("../src/application/decomposition/eigen.rs"),
    ),
    (
        "full_piv_lu",
        include_str!("../src/application/decomposition/full_piv_lu.rs"),
    ),
    (
        "hessenberg",
        include_str!("../src/application/decomposition/hessenberg.rs"),
    ),
    ("lu", include_str!("../src/application/decomposition/lu.rs")),
    ("qr", include_str!("../src/application/decomposition/qr.rs")),
    (
        "schur",
        include_str!("../src/application/decomposition/schur.rs"),
    ),
    (
        "svd",
        include_str!("../src/application/decomposition/svd.rs"),
    ),
    (
        "udu",
        include_str!("../src/application/decomposition/udu.rs"),
    ),
];

#[test]
fn decomposition_heap_readbacks_are_provider_owned() {
    for (name, source) in DECOMPOSITION_SOURCES {
        assert_direct_downloads_are_one_word_stack_reads("CUDA", name, source);
        assert!(
            source.contains("download_owned("),
            "CUDA {name} must retain provider-owned readback"
        );
    }
}

fn assert_direct_downloads_are_one_word_stack_reads(provider: &str, name: &str, source: &str) {
    for line in source.lines().map(str::trim) {
        match line {
            "device.download(&status, &mut status_host)?;" => assert!(
                source.contains("let mut status_host = [0_u32; 1];"),
                "{provider} {name} status readback must target a one-word stack array"
            ),
            "device.download(&rank, &mut rank_host)?;" => assert!(
                source.contains("let mut rank_host = [0_u32; 1];"),
                "{provider} {name} rank readback must target a one-word stack array"
            ),
            line if line.contains("device.download(") => {
                panic!("{provider} {name} heap readback must use download_owned: {line}")
            }
            _ => {}
        }
    }
}

#[test]
fn cuda_satisfies_the_decomposition_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA decomposition conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA decomposition conformance requires a physical device: {error}"),
    };
    assert_decomposition_contract(&device, &CudaDecompositionOps);
}
