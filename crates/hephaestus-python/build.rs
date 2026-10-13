//! Python extension and embedded-test link contracts.
//!
//! Cargo tests use PyO3's interpreter library discovery. Maturin >= 1.9.4
//! selects extension-only linking through `PYO3_BUILD_EXTENSION_MODULE`;
//! Apple's cdylib then resolves Python symbols from the loading interpreter.
//! PyO3's build helper owns the target-specific cdylib linker arguments.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
