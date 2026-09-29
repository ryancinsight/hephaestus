//! macOS link behavior for the `pyhephaestus` Python extension cdylib.
//!
//! The cdylib is a Python extension module: it must never link a Python
//! interpreter library, because the interpreter that loads it supplies the
//! Python symbols at load time. On Apple's linker that contract is spelled
//! `-undefined dynamic_lookup`. Leaving the behavior unpinned (dependent on
//! pyo3's feature resolution and the build environment) risks the same
//! intermittent `Undefined symbols: _PyBaseObject_Type` link failure the
//! consus cdylib exhibited on its macOS CI runner. This repo's CI never
//! links the Python cdylib on macOS — the `metal.yml` job checks only the
//! Metal crate — so the macOS link happens in the release wheel build,
//! where a flake is most expensive.
//!
//! Emitting the flag from here pins it for every build path that links this
//! cdylib. It is emitted only for Apple targets, so other platforms and
//! dependency builds are untouched.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }
    // `-undefined dynamic_lookup` defers Python symbol resolution to load
    // time, which is the defining property of an extension module.
    // `rustc-cdylib-link-arg` targets exactly this crate's cdylib artifact.
    println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
}
