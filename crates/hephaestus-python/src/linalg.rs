//! Dense linear-algebra products on `Array`: matmul, dot, trace, kron,
//! and batched matmul, plus their module-level function forms.

use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyArray {
    fn matmul(&self, py: Python<'_>, other: &PyArray) -> PyResult<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(PyValueError::new_err("matmul requires 2D arrays"));
        }
        if self.shape[1] != other.shape[0] {
            return Err(PyValueError::new_err(format!(
                "matmul shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let layout_a =
            Layout::c_contiguous([m, k]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let layout_b =
            Layout::c_contiguous([k, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf_a = self.buffer.clone();
        let buf_b = other.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf_a, &buf_b) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(a), BackendBuffer::Wgpu(b)) => {
                    hephaestus_wgpu::matmul(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_wgpu::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(a), BackendBuffer::Cuda(b)) => {
                    hephaestus_cuda::matmul(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_cuda::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "matmul operands belong to different backends".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![m, n],
        })
    }

    fn dot(&self, py: Python<'_>, other: &PyArray) -> PyResult<Self> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(PyValueError::new_err("dot requires 1D arrays"));
        }
        if self.shape[0] != other.shape[0] {
            return Err(PyValueError::new_err(format!(
                "dot shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        let len = self.shape[0];
        let layout_a =
            Layout::c_contiguous([len]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let layout_b =
            Layout::c_contiguous([len]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf_a = self.buffer.clone();
        let buf_b = other.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf_a, &buf_b) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(a), BackendBuffer::Wgpu(b)) => {
                    hephaestus_wgpu::dot(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_wgpu::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(a), BackendBuffer::Cuda(b)) => {
                    hephaestus_cuda::dot(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_cuda::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "dot operands belong to different backends".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn trace(&self, py: Python<'_>) -> PyResult<Self> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err("trace requires a 2D array"));
        }
        if self.shape[0] != self.shape[1] {
            return Err(PyValueError::new_err("trace requires a square matrix"));
        }
        let n = self.shape[0];
        let layout =
            Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    hephaestus_wgpu::trace(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    hephaestus_cuda::trace(
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
            shape: vec![1],
        })
    }

    /// Kronecker product `kron(self, other)` of two 2-D arrays.
    fn kron(&self, py: Python<'_>, other: &PyArray) -> PyResult<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(PyValueError::new_err("kron requires 2D arrays"));
        }
        let (r1, c1) = (self.shape[0], self.shape[1]);
        let (r2, c2) = (other.shape[0], other.shape[1]);
        let layout_a =
            Layout::c_contiguous([r1, c1]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let layout_b =
            Layout::c_contiguous([r2, c2]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf_a = self.buffer.clone();
        let buf_b = other.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf_a, &buf_b) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(a), BackendBuffer::Wgpu(b)) => {
                    hephaestus_wgpu::kron(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_wgpu::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(a), BackendBuffer::Cuda(b)) => {
                    hephaestus_cuda::kron(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_cuda::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "kron operands belong to different backends".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![r1 * r2, c1 * c2],
        })
    }

    /// Batched matrix multiply: `[batch, m, k] @ [batch, k, n] -> [batch, m, n]`.
    fn batched_matmul(&self, py: Python<'_>, other: &PyArray) -> PyResult<Self> {
        if self.shape.len() != 3 || other.shape.len() != 3 {
            return Err(PyValueError::new_err(
                "batched_matmul requires 3D arrays [batch, m, k]",
            ));
        }
        let (batch, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
        let (batch_b, k_b, n) = (other.shape[0], other.shape[1], other.shape[2]);
        if batch != batch_b || k != k_b {
            return Err(PyValueError::new_err(format!(
                "batched_matmul shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        let layout_a = Layout::c_contiguous([batch, m, k])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let layout_b = Layout::c_contiguous([batch, k, n])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf_a = self.buffer.clone();
        let buf_b = other.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf_a, &buf_b) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(a), BackendBuffer::Wgpu(b)) => {
                    hephaestus_wgpu::batched_matmul(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_wgpu::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(a), BackendBuffer::Cuda(b)) => {
                    hephaestus_cuda::batched_matmul(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer: a,
                            layout: &layout_a,
                        },
                        hephaestus_cuda::StridedOperand {
                            buffer: b,
                            layout: &layout_b,
                        },
                    )
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "batched_matmul operands belong to different backends".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![batch, m, n],
        })
    }
}

// ── Module-level function forms ──

#[pyfunction]
#[pyo3(name = "matmul")]
pub(crate) fn matmul_py(py: Python<'_>, a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    a.matmul(py, b)
}

#[pyfunction]
#[pyo3(name = "dot")]
pub(crate) fn dot_py(py: Python<'_>, a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    a.dot(py, b)
}

#[pyfunction]
#[pyo3(name = "trace")]
pub(crate) fn trace_py(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.trace(py)
}
