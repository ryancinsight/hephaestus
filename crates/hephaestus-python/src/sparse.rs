//! Python-visible `SparseMatrix` (CSR) class and sparse products
//! (spmv, spmm, batched spmv).

use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendCsrMatrix, BackendDevice};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper around a GPU-resident CSR matrix.
#[pyclass(name = "SparseMatrix")]
pub struct PyCsrMatrix {
    inner: BackendCsrMatrix,
    device: BackendDevice,
}

#[pymethods]
impl PyCsrMatrix {
    /// Create a SparseMatrix from a dense PyArray on the GPU.
    #[staticmethod]
    fn from_dense(py: Python<'_>, arr: &PyArray) -> PyResult<Self> {
        if arr.shape.len() != 2 {
            return Err(PyValueError::new_err(
                "SparseMatrix can only be constructed from a 2D array",
            ));
        }
        let [rows, cols] = [arr.shape[0], arr.shape[1]];
        let layout =
            Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let device = arr.device.clone();
        let buffer = arr.buffer.clone();
        let len = arr.buffer.len();

        let host_data = py
            .allow_threads(move || {
                let mut host_data = vec![0.0f32; len];
                device
                    .download_f32(&buffer, &mut host_data)
                    .map(|_| host_data)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let view = leto::ArrayView2::new(layout, &host_data);
        let cpu_csr = leto_ops::CsrMatrix::from_dense(&view);

        let inner = match &arr.device {
            BackendDevice::Wgpu(device) => {
                hephaestus_wgpu::GpuCsrMatrix::from_cpu(device, &cpu_csr)
                    .map(BackendCsrMatrix::Wgpu)
            }
            BackendDevice::Cuda(device) => {
                hephaestus_cuda::GpuCsrMatrix::from_cpu(device, &cpu_csr)
                    .map(|matrix| BackendCsrMatrix::Cuda(Arc::new(matrix)))
            }
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner,
            device: arr.device.clone(),
        })
    }

    /// Reconstruct the dense matrix as a PyArray.
    fn to_dense(&self, py: Python<'_>) -> PyResult<PyArray> {
        let device = self.device.clone();
        let inner = self.inner.clone();
        let (rows, cols) = inner.shape();

        let cpu_csr = py
            .allow_threads(move || match (&device, &inner) {
                (BackendDevice::Wgpu(device), BackendCsrMatrix::Wgpu(matrix)) => {
                    matrix.to_cpu(device)
                }
                (BackendDevice::Cuda(device), BackendCsrMatrix::Cuda(matrix)) => {
                    matrix.to_cpu(device)
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "sparse matrix belongs to a different backend".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let cpu_dense = cpu_csr.to_dense();
        let dense_buf = self
            .device
            .upload_f32(leto::Storage::as_slice(cpu_dense.storage()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyArray {
            buffer: dense_buf,
            device: self.device.clone(),
            shape: vec![rows, cols],
        })
    }

    /// shape of the matrix: (rows, cols)
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// number of non-zero elements
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }
}

#[pyfunction]
pub(crate) fn spmv(py: Python<'_>, a: &PyCsrMatrix, x: &PyArray) -> PyResult<PyArray> {
    let device = a.device.clone();
    let inner_a = a.inner.clone();
    let buf_x = x.buffer.clone();

    let out_buf = py
        .allow_threads(move || match (&device, &inner_a, &buf_x) {
            (
                BackendDevice::Wgpu(device),
                BackendCsrMatrix::Wgpu(matrix),
                BackendBuffer::Wgpu(x),
            ) => hephaestus_wgpu::spmv(device, matrix, x).map(BackendBuffer::Wgpu),
            (
                BackendDevice::Cuda(device),
                BackendCsrMatrix::Cuda(matrix),
                BackendBuffer::Cuda(x),
            ) => hephaestus_cuda::spmv(device, matrix, x)
                .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "spmv operands belong to different backends".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let (rows, _) = a.inner.shape();
    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![rows],
    })
}

#[pyfunction]
pub(crate) fn spmm(py: Python<'_>, a: &PyCsrMatrix, b: &PyArray) -> PyResult<PyArray> {
    if b.shape.len() != 2 {
        return Err(PyValueError::new_err(
            "spmm requires a 2D dense matrix as the right-hand side",
        ));
    }
    let device = a.device.clone();
    let inner_a = a.inner.clone();
    let buf_b = b.buffer.clone();
    let [b_rows, bcols] = [b.shape[0], b.shape[1]];
    let layout_b =
        Layout::c_contiguous([b_rows, bcols]).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out_buf = py
        .allow_threads(move || match (&device, &inner_a, &buf_b) {
            (
                BackendDevice::Wgpu(device),
                BackendCsrMatrix::Wgpu(matrix),
                BackendBuffer::Wgpu(buffer),
            ) => {
                let op_b = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout_b,
                };
                hephaestus_wgpu::spmm(device, matrix, &op_b).map(BackendBuffer::Wgpu)
            }
            (
                BackendDevice::Cuda(device),
                BackendCsrMatrix::Cuda(matrix),
                BackendBuffer::Cuda(buffer),
            ) => {
                let op_b = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout_b,
                };
                hephaestus_cuda::spmm(device, matrix, &op_b)
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "spmm operands belong to different backends".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let (rows, _) = a.inner.shape();
    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![rows, bcols],
    })
}

#[pyfunction]
pub(crate) fn spmv_many(py: Python<'_>, a: &PyCsrMatrix, x_batch: &PyArray) -> PyResult<PyArray> {
    if x_batch.shape.len() != 2 {
        return Err(PyValueError::new_err(
            "spmv_many requires a 2D dense matrix of RHS vectors",
        ));
    }
    let device = a.device.clone();
    let inner_a = a.inner.clone();
    let buf_x = x_batch.buffer.clone();
    let [x_rows, xcols] = [x_batch.shape[0], x_batch.shape[1]];
    let layout_x =
        Layout::c_contiguous([x_rows, xcols]).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out_buf = py
        .allow_threads(move || match (&device, &inner_a, &buf_x) {
            (
                BackendDevice::Wgpu(device),
                BackendCsrMatrix::Wgpu(matrix),
                BackendBuffer::Wgpu(buffer),
            ) => {
                let op_x = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout_x,
                };
                hephaestus_wgpu::spmv_many(device, matrix, &op_x).map(BackendBuffer::Wgpu)
            }
            (
                BackendDevice::Cuda(device),
                BackendCsrMatrix::Cuda(matrix),
                BackendBuffer::Cuda(buffer),
            ) => {
                let op_x = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout_x,
                };
                hephaestus_cuda::spmv_many(device, matrix, &op_x)
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "spmv_many operands belong to different backends".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let (rows, _) = a.inner.shape();
    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![rows, xcols],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::PyDevice;
    use crate::test_support::prepare_python;

    #[test]
    fn test_py_sparse_matrix_roundtrip_spmv_spmm() {
        prepare_python();
        Python::with_gil(|py| {
            let device = PyDevice::new(None).unwrap();

            // Create a 3x3 matrix:
            // [ 2.0  0.0 -1.0 ]
            // [ 0.0  3.0  0.0 ]
            // [ 0.0  0.0  4.0 ]
            let dense_data = vec![2.0f32, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
            let dense_arr = PyArray::new(py, dense_data.clone(), &device)
                .unwrap()
                .reshape(vec![3, 3])
                .unwrap();

            let sparse = PyCsrMatrix::from_dense(py, &dense_arr).unwrap();
            assert_eq!(sparse.shape(), (3, 3));
            assert_eq!(sparse.nnz(), 4);

            // test to_dense
            let dense_reconstructed = sparse.to_dense(py).unwrap();
            assert_eq!(dense_reconstructed.tolist(py).unwrap(), dense_data);

            // test spmv: y = A * x, x = [1.0, 2.0, 3.0]
            let x = PyArray::new(py, vec![1.0f32, 2.0, 3.0], &device).unwrap();
            let y = spmv(py, &sparse, &x).unwrap();
            assert_eq!(y.tolist(py).unwrap(), vec![-1.0f32, 6.0, 12.0]);

            // test spmm: C = A * B, B = [ 1.0  2.0 ]
            //                            [ 3.0  4.0 ]
            //                            [ 5.0  6.0 ]
            let b = PyArray::new(py, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
                .unwrap()
                .reshape(vec![3, 2])
                .unwrap();
            let c = spmm(py, &sparse, &b).unwrap();
            assert_eq!(
                c.tolist(py).unwrap(),
                vec![-3.0f32, -2.0, 9.0, 12.0, 20.0, 24.0]
            );

            let many = spmv_many(py, &sparse, &b).unwrap();
            assert_eq!(
                many.tolist(py).unwrap(),
                vec![-3.0f32, -2.0, 9.0, 12.0, 20.0, 24.0]
            );
        });
    }
}
