//! Elementwise operations on `Array`: unary transcendentals, binary
//! arithmetic dunders (array-array and array-scalar), and their
//! module-level function forms.

mod binary;
mod unary;

use crate::array::PyArray;
use pyo3::prelude::*;

// ── Module-level function forms ──

#[pyfunction]
pub(crate) fn add(py: Python<'_>, a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__add__(py, b)
}

#[pyfunction]
pub(crate) fn sub(py: Python<'_>, a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__sub__(py, b)
}

#[pyfunction]
pub(crate) fn mul(py: Python<'_>, a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__mul__(py, b)
}

#[pyfunction]
pub(crate) fn div(py: Python<'_>, a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__truediv__(py, b)
}

#[pyfunction]
pub(crate) fn pow(py: Python<'_>, a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__pow__(py, b, None)
}

#[pyfunction]
pub(crate) fn exp(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.exp(py)
}

#[pyfunction]
pub(crate) fn log(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.log(py)
}

#[pyfunction]
pub(crate) fn sin(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.sin(py)
}

#[pyfunction]
pub(crate) fn cos(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.cos(py)
}

#[pyfunction]
pub(crate) fn sqrt(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.sqrt(py)
}

#[pyfunction]
pub(crate) fn abs(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.abs(py)
}

#[pyfunction]
pub(crate) fn neg(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.neg(py)
}
