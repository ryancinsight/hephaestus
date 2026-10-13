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
            assert!(
                absent,
                "requested missing library is classified as absent: {source}"
            );
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
    #[cfg(target_os = "macos")]
    let name = "/usr/lib/libSystem.B.dylib";
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
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

#[cfg(target_os = "macos")]
#[test]
fn missing_dependency_is_a_driver_fault() {
    use std::path::PathBuf;
    use std::process::Command;

    #[derive(Debug)]
    struct LibraryDirectory(PathBuf);
    impl Drop for LibraryDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = LibraryDirectory(std::env::temp_dir().join(format!(
        "hephaestus-loader-dependency-{}",
        std::process::id()
    )));
    std::fs::create_dir(&directory.0).expect("unique library fixture directory");
    let dependency = directory.0.join("dependency.dylib");
    let requested = directory.0.join("requested.dylib");
    let dependency_source = directory.0.join("dependency.c");
    let requested_source = directory.0.join("requested.c");
    {
        let _phase = tracing::info_span!("arrange", contract = "real_loader_dependency").entered();
        std::fs::write(
            &dependency_source,
            "int dependency_value(int input) { return input * input + 1; }\n",
        )
        .expect("dependency source");
        std::fs::write(
            &requested_source,
            "int dependency_value(int);\nint loader_value(int input) { return dependency_value(input) + input; }\n",
        )
        .expect("requested image source");
        for (source, output, dependencies) in [
            (&dependency_source, &dependency, Vec::new()),
            (&requested_source, &requested, vec![&dependency]),
        ] {
            let compiled = Command::new("cc")
                .arg("-dynamiclib")
                .arg(source)
                .args(dependencies)
                .arg("-o")
                .arg(output)
                .output()
                .expect("native compiler");
            assert!(
                compiled.status.success(),
                "native fixture compilation: {}",
                String::from_utf8_lossy(&compiled.stderr)
            );
        }
        // SAFETY: these owned images contain the audited C functions above.
        // The symbol signature is C int -> C int, and the library lives until
        // every call finishes. Inputs bound all polynomial intermediates.
        let library = unsafe { libloading::Library::new(&requested) }.expect("complete image");
        // SAFETY: loader_value is the owned image's C int -> C int export;
        // its symbol borrow cannot outlive the retained library.
        let function = unsafe {
            library
                .get::<unsafe extern "C" fn(core::ffi::c_int) -> core::ffi::c_int>(b"loader_value")
        }
        .expect("polynomial export");
        for (input, expected) in [(2, 7), (3, 13), (5, 31)] {
            // SAFETY: the validated export is live and every input is in the
            // bounded exact-integer domain established above.
            assert_eq!(unsafe { function(input) }, expected);
        }
    }
    std::fs::remove_file(&dependency).expect("remove owned dependency image");
    let error = {
        let _phase = tracing::info_span!("act", contract = "missing_dependency").entered();
        Driver::load(requested.to_str().expect("fixture path"))
            .expect_err("present image cannot load without its dependency")
    };
    let _phase = tracing::info_span!("assert", contract = "fault_identity").entered();
    match error.report() {
        hephaestus_core::HephaestusError::DeviceUnavailable { message } => {
            assert!(message.contains(requested.to_str().expect("requested path")));
            assert!(message.contains(dependency.to_str().expect("dependency path")));
        }
        other => panic!("dependent-image fault was hidden as adapter absence: {other}"),
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
