#![deny(missing_docs)]
//! # pyhephaestus
//!
//! Thin PyO3 binding surface over the hephaestus GPU compute backends
//! (`hephaestus-wgpu`, `hephaestus-cuda`).
//!
//! The binding layer marshals Python/NumPy values to device buffers,
//! dispatches to backend kernels with the GIL released
//! (`Python::detach`), and maps `HephaestusError` to Python
//! exceptions. It holds no domain logic: matrix mathematics lives in
//! `hephaestus-core` and the backend crates.
//!
//! Module layout (one leaf module per operation family):
//! - `backend` — backend device/buffer enums and dispatch macros
//! - `device` / `array` — Python-visible `Device` and `Array` classes
//! - `elementwise`, `reduction`, `scan` — pointwise ops, reductions, scans
//! - `linalg`, `matfunc` — dense products and matrix functions
//! - `decomposition`, `spectral` — factorisations and eigen/SVD routines
//! - `sparse` — CSR `SparseMatrix` class and sparse products
//! - `random` — seeded RNG initialisers

use pyo3::prelude::*;

mod array;
mod backend;
mod decomposition;
mod device;
mod elementwise;
mod linalg;
mod matfunc;
mod module;
mod random;
mod reduction;
mod scan;
mod sparse;
mod spectral;
#[cfg(test)]
pub(crate) mod test_support;

/// PyHephaestus extension module definition.
#[pymodule]
fn pyhephaestus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    module::register(m)
}
