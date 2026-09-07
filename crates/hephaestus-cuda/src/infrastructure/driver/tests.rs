//! Failure contracts exercised against the platform loader, without driver mocks.

use super::loader::{Driver, LoadError};

#[test]
fn absent_library_preserves_its_identity() {
    let name = "atlas-cuda-driver-that-does-not-exist";
    let error = Driver::load(name).expect_err("nonexistent library cannot load");
    match error {
        LoadError::Library {
            name: actual,
            source,
            absent,
        } => {
            assert_eq!(actual, name);
            assert!(absent, "requested missing library is classified as absent");
            assert!(
                !source.to_string().is_empty(),
                "loader diagnostic is preserved"
            );
        }
        other => panic!("wrong failure category: {other:?}"),
    }
}

#[test]
fn required_symbol_failure_preserves_export_identity() {
    #[cfg(target_os = "windows")]
    let name = "kernel32.dll";
    #[cfg(not(target_os = "windows"))]
    let name = "libc.so.6";
    let error = Driver::load(name).expect_err("system library does not export CUDA");
    match error {
        LoadError::Symbol { name, source } => {
            assert_eq!(name, "cuCtxCreate_v2");
            assert!(
                !source.to_string().is_empty(),
                "symbol diagnostic is preserved"
            );
        }
        other => panic!("wrong failure category: {other:?}"),
    }
}

#[test]
fn initialization_statuses_preserve_failure_categories() {
    for (status, name) in [
        (34, "CUDA_ERROR_STUB_LIBRARY"),
        (100, "CUDA_ERROR_NO_DEVICE"),
    ] {
        match LoadError::Initialize(status).report() {
            hephaestus_core::HephaestusError::AdapterUnavailable { message } => {
                assert_eq!(message, format!("cuInit -> {status} ({name})"));
            }
            error => panic!("wrong absent-driver/device category: {error}"),
        }
    }
    for status in [1, 3, 35, 36, 46, 803, 999] {
        match LoadError::Initialize(status).report() {
            hephaestus_core::HephaestusError::DeviceUnavailable { message } => {
                assert_eq!(message, format!("cuInit -> {status}"))
            }
            error => panic!("driver fault hidden as absence: {error}"),
        }
    }
}

#[test]
fn present_non_library_is_a_driver_fault() {
    #[cfg(target_os = "windows")]
    let name = "drivers\\etc\\hosts";
    #[cfg(not(target_os = "windows"))]
    let name = "/etc/hosts";
    let error = Driver::load(name).expect_err("system hosts file is not a library");
    match error.report() {
        hephaestus_core::HephaestusError::DeviceUnavailable { message } => {
            assert!(
                message.contains(name),
                "fault retains requested library identity"
            );
        }
        other => panic!("loader failure was hidden as absence: {other}"),
    }
}
