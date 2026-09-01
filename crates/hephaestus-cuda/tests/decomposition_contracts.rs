//! CUDA instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

#![cfg(all(feature = "cuda", feature = "decomposition"))]
use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_cuda::{CudaDecompositionOps, CudaDevice, split_packed_lu};
use std::sync::OnceLock;

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
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    assert_decomposition_contract(&device, &CudaDecompositionOps);
}

fn cuda_device_or_skip() -> Option<CudaDevice> {
    static DEVICE: OnceLock<Option<CudaDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| match CudaDevice::try_default() {
            Ok(device) => Some(device),
            Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
                eprintln!("skip CUDA decomposition test: device unavailable ({error})");
                None
            }
            Err(error) => {
                panic!("CUDA decomposition tests require a physical device: {error}")
            }
        })
        .clone()
}

fn packed_factor(n: usize) -> Vec<f32> {
    (0..n * n)
        .map(|index| {
            let row = index / n;
            let column = index % n;
            let ordinal = u16::try_from(row * 32 + column).expect("test matrix fits u16");
            f32::from(ordinal) + 0.5
        })
        .collect()
}

#[test]
fn cuda_split_packed_lu_matches_host_oracle_at_block_boundaries() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    for n in [1, 2, 16, 17] {
        let host_packed = packed_factor(n);
        let (expected_lower, expected_upper) =
            hephaestus_core::split_packed_lu(&host_packed, n).expect("host split");
        let packed = device.upload(&host_packed).expect("packed upload");
        let (lower, upper) = split_packed_lu(&device, &packed, n).expect("CUDA split");
        let actual_lower = device.download_owned(&lower).expect("lower download");
        let actual_upper = device.download_owned(&upper).expect("upper download");

        assert_eq!(actual_lower, expected_lower, "lower factor at n={n}");
        assert_eq!(actual_upper, expected_upper, "upper factor at n={n}");
    }
}

#[test]
fn cuda_split_packed_lu_handles_empty_and_rejects_length_mismatch() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let empty = device.alloc_zeroed::<f32>(0).expect("empty buffer");
    let (lower, upper) = split_packed_lu(&device, &empty, 0).expect("empty split");
    assert_eq!(lower.len(), 0);
    assert_eq!(upper.len(), 0);

    let malformed = device.alloc_zeroed::<f32>(5).expect("malformed buffer");
    let error = split_packed_lu(&device, &malformed, 3).expect_err("length mismatch");
    assert!(matches!(
        error,
        HephaestusError::LengthMismatch {
            host_len: 9,
            device_len: 5,
        }
    ));
}

#[test]
fn cuda_split_packed_lu_rejects_foreign_device_before_mutation() {
    let Some(owner) = cuda_device_or_skip() else {
        return;
    };
    let dispatch = CudaDevice::try_default().expect("second CUDA context");
    let original = packed_factor(2);
    let packed = owner.upload(&original).expect("owner upload");

    let error = split_packed_lu(&dispatch, &packed, 2).expect_err("foreign buffer rejection");
    assert!(matches!(
        error,
        HephaestusError::InvalidConfiguration { message }
            if message == "packed LU buffer must belong to the dispatch device"
    ));
    assert_eq!(
        owner
            .download_owned(&packed)
            .expect("unchanged packed readback"),
        original
    );
}
