//! Extension-module registration: adds the Python-visible classes and
//! functions to `pyhephaestus`.
//!
//! Kept out of the crate root so `lib.rs` stays a thin facade around the
//! `#[pymodule]` entry point; PyO3 requires that entry point itself to live
//! at the crate root.

use pyo3::prelude::*;

/// Register the Python-visible classes and functions on `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::device::PyDevice>()?;
    m.add_class::<crate::array::PyArray>()?;
    m.add_class::<crate::sparse::PyCsrMatrix>()?;

    m.add_function(wrap_pyfunction!(crate::elementwise::add, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::sub, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::mul, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::div, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::pow, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::exp, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::log, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::sin, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::cos, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::abs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::elementwise::neg, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::sum, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::min, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::max, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::mean, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::dot_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::trace_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::norm_l1_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::norm_l2_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reduction::norm_max_py, m)?)?;

    m.add_function(wrap_pyfunction!(crate::decomposition::cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::lu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::hessenberg, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::full_piv_lu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::bidiagonalize, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::qr, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::col_piv_qr, m)?)?;
    m.add_function(wrap_pyfunction!(crate::spectral::svd, m)?)?;
    m.add_function(wrap_pyfunction!(crate::spectral::symmetric_eigen, m)?)?;
    m.add_function(wrap_pyfunction!(crate::spectral::singular_values, m)?)?;
    m.add_function(wrap_pyfunction!(crate::spectral::schur, m)?)?;
    m.add_function(wrap_pyfunction!(crate::decomposition::bunch_kaufman, m)?)?;
    m.add_function(wrap_pyfunction!(crate::matfunc::matexp, m)?)?;
    m.add_function(wrap_pyfunction!(crate::matfunc::pinv, m)?)?;
    m.add_function(wrap_pyfunction!(crate::spectral::eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(crate::sparse::spmv, m)?)?;
    m.add_function(wrap_pyfunction!(crate::sparse::spmv_many, m)?)?;
    m.add_function(wrap_pyfunction!(crate::sparse::spmm, m)?)?;
    m.add_function(wrap_pyfunction!(crate::random::uniform_with_seed, m)?)?;
    m.add_function(wrap_pyfunction!(crate::random::normal_with_seed, m)?)?;

    Ok(())
}
