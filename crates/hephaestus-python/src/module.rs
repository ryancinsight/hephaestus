//! PyHephaestus extension module registration.

use pyo3::prelude::*;

use crate::{
    array, decomposition, device, elementwise, linalg, matfunc, random, reduction, sparse, spectral,
};

/// PyHephaestus extension module definition.
#[pymodule]
fn pyhephaestus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<device::PyDevice>()?;
    m.add_class::<array::PyArray>()?;
    m.add_class::<sparse::PyCsrMatrix>()?;

    m.add_function(wrap_pyfunction!(elementwise::add, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::sub, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::mul, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::div, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::pow, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::exp, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::log, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::sin, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::cos, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::abs, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise::neg, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::sum, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::min, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::max, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::mean, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::dot_py, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::trace_py, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::norm_l1_py, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::norm_l2_py, m)?)?;
    m.add_function(wrap_pyfunction!(reduction::norm_max_py, m)?)?;

    m.add_function(wrap_pyfunction!(decomposition::cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::lu, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::hessenberg, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::full_piv_lu, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::bidiagonalize, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::qr, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::col_piv_qr, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::svd, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::symmetric_eigen, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::singular_values, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::schur, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::bunch_kaufman, m)?)?;
    m.add_function(wrap_pyfunction!(matfunc::matexp, m)?)?;
    m.add_function(wrap_pyfunction!(matfunc::pinv, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::spmv, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::spmv_many, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::spmm, m)?)?;
    m.add_function(wrap_pyfunction!(random::uniform_with_seed, m)?)?;
    m.add_function(wrap_pyfunction!(random::normal_with_seed, m)?)?;

    Ok(())
}
