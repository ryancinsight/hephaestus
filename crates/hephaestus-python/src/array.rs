//! Python-visible `Array` class: GPU-resident `f32` array with host
//! transfer, shape bookkeeping, and shared shape-validation helpers.
//! Operation families (elementwise, reduction, scan, linalg, ...) extend
//! `PyArray` from their own modules via `multiple-pymethods`.

use crate::backend::{BackendBuffer, BackendDevice};
use crate::device::PyDevice;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Python wrapper around a GPU-resident WgpuBuffer<f32>.
#[pyclass(name = "Array")]
pub struct PyArray {
    pub(crate) buffer: BackendBuffer,
    pub(crate) device: BackendDevice,
    #[pyo3(get)]
    pub shape: Vec<usize>,
}

impl PyArray {
    /// Validate that this array is a square 2-D matrix, returning its dimension.
    ///
    /// Pure-Rust helper (not exposed to Python); shared by the square-matrix
    /// linalg methods (`det`, `matexp`, `matpow`).
    pub(crate) fn require_square(&self, op: &str) -> PyResult<usize> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err(format!("{op} requires a 2D array")));
        }
        if self.shape[0] != self.shape[1] {
            return Err(PyValueError::new_err(format!(
                "{op} requires a square matrix, got shape {:?}",
                self.shape
            )));
        }
        Ok(self.shape[0])
    }

    /// Validate a 2-D array and an `axis` in `{0, 1}`, returning `(rows, cols)`.
    ///
    /// Pure-Rust helper shared by the axis reductions and `cumsum`.
    pub(crate) fn require_axis_2d(&self, op: &str, axis: usize) -> PyResult<(usize, usize)> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err(format!("{op} requires a 2D array")));
        }
        if axis > 1 {
            return Err(PyValueError::new_err(format!(
                "{op} axis must be 0 or 1, got {axis}"
            )));
        }
        Ok((self.shape[0], self.shape[1]))
    }
}

#[pymethods]
impl PyArray {
    /// Upload a python list/iterable of floats to the GPU.
    #[new]
    #[pyo3(signature = (data, device))]
    pub(crate) fn new(py: Python<'_>, data: Vec<f32>, device: &PyDevice) -> PyResult<Self> {
        let len = data.len();
        let dev = device.inner.clone();
        let buffer = py
            .allow_threads(move || dev.upload_f32(&data))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
            shape: vec![len],
        })
    }

    /// Allocate a zeroed array of a given length on the GPU.
    #[staticmethod]
    fn zeros(len: usize, device: &PyDevice) -> PyResult<Self> {
        let buffer = device
            .inner
            .alloc_zeroed_f32(len)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
            shape: vec![len],
        })
    }

    /// Create an Array from a contiguous NumPy array.
    #[staticmethod]
    fn from_numpy(
        py: Python<'_>,
        arr: PyReadonlyArray1<'_, f32>,
        device: &PyDevice,
    ) -> PyResult<Self> {
        let slice = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let len = slice.len();
        let dev = device.inner.clone();
        let buffer = py
            .allow_threads(move || dev.upload_f32(slice))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
            shape: vec![len],
        })
    }

    /// Reshape the array to a new shape.
    pub(crate) fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let total: usize = shape.iter().product();
        if total != self.buffer.len() {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape {:?}",
                self.buffer.len(),
                shape
            )));
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            device: self.device.clone(),
            shape,
        })
    }

    /// Download array data to a Python list.
    pub(crate) fn tolist(&self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let len = self.buffer.len();
        let host_data = py
            .allow_threads(move || {
                let mut host_data = vec![0.0f32; len];
                dev.download_f32(&buf, &mut host_data).map(|_| host_data)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(host_data)
    }

    /// Download array data to a NumPy 1D array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let len = self.buffer.len();
        let host_data = py
            .allow_threads(move || {
                let mut host_data = vec![0.0f32; len];
                dev.download_f32(&buf, &mut host_data).map(|_| host_data)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(host_data.to_pyarray(py))
    }

    /// Get the length of the array.
    #[getter]
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::prepare_python;
    use numpy::PyArrayMethods;

    #[test]
    fn test_py_array_tolist_and_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let device = PyDevice::new(None).unwrap();
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let py_arr = PyArray::new(py, data.clone(), &device).unwrap();
            assert_eq!(py_arr.tolist(py).unwrap(), data);

            let np_arr = py_arr.to_numpy(py).unwrap();
            assert_eq!(np_arr.readonly().as_slice().unwrap(), &data[..]);
        });
    }
}
