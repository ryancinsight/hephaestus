//! Matrix functions and inverses on `Array`: determinant, matrix
//! exponential/power, numerical rank, and pseudo-inverse, plus their
//! module-level function forms.

use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyArray {
    /// Determinant of a square matrix (returned as a length-1 array).
    fn det(&self, py: Python<'_>) -> PyResult<Self> {
        let n = self.require_square("det")?;
        let layout =
            Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => hephaestus_wgpu::det(
                    device,
                    hephaestus_wgpu::StridedOperand {
                        buffer,
                        layout: &layout,
                    },
                )
                .map(BackendBuffer::Wgpu),
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => hephaestus_cuda::det(
                    device,
                    hephaestus_cuda::StridedOperand {
                        buffer,
                        layout: &layout,
                    },
                )
                .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer))),
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "array buffer belongs to a different backend".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    /// Matrix exponential `expm(A)` of a square matrix.
    fn matexp(&self, py: Python<'_>) -> PyResult<Self> {
        let n = self.require_square("matexp")?;
        let layout =
            Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    hephaestus_wgpu::matexp(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    hephaestus_cuda::matexp(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "array buffer belongs to a different backend".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![n, n],
        })
    }

    /// Integer matrix power `A**exponent` of a square matrix.
    fn matpow(&self, py: Python<'_>, exponent: u32) -> PyResult<Self> {
        let n = self.require_square("matpow")?;
        let layout =
            Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    hephaestus_wgpu::matpow(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                        exponent,
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    hephaestus_cuda::matpow(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                        exponent,
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "array buffer belongs to a different backend".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![n, n],
        })
    }

    /// Numerical rank of a 2-D matrix (default tolerance), returned as an int.
    fn matrix_rank(&self, py: Python<'_>) -> PyResult<usize> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err("matrix_rank requires a 2D array"));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let layout =
            Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        py.allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                hephaestus_wgpu::matrix_rank(
                    device,
                    hephaestus_wgpu::StridedOperand {
                        buffer,
                        layout: &layout,
                    },
                )
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                hephaestus_cuda::matrix_rank(
                    device,
                    hephaestus_cuda::StridedOperand {
                        buffer,
                        layout: &layout,
                    },
                )
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Moore-Penrose pseudo-inverse of a 2-D matrix (shape `[cols, rows]`).
    fn pinv(&self, py: Python<'_>) -> PyResult<Self> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err("pinv requires a 2D array"));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let layout =
            Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    hephaestus_wgpu::pinv(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    hephaestus_cuda::pinv(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "array buffer belongs to a different backend".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![cols, rows],
        })
    }
}

// ── Module-level function forms ──

#[pyfunction]
pub(crate) fn matexp(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("matexp requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let out_buf = py
        .allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => hephaestus_wgpu::matexp(
                device,
                hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                },
            )
            .map(BackendBuffer::Wgpu),
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => hephaestus_cuda::matexp(
                device,
                hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                },
            )
            .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![n, n],
    })
}

#[pyfunction]
pub(crate) fn pinv(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("pinv requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let out_buf = py
        .allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => hephaestus_wgpu::pinv(
                device,
                hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                },
            )
            .map(BackendBuffer::Wgpu),
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => hephaestus_cuda::pinv(
                device,
                hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                },
            )
            .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![cols, rows],
    })
}
