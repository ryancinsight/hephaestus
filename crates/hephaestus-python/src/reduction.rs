//! Reductions on `Array`: full reductions, axis reductions, and norms,
//! plus their module-level function forms.

use crate::array::PyArray;
use crate::backend::{
    BackendBuffer, BackendDevice, backend_norm, backend_reduction, backend_scalar,
};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

impl PyArray {
    /// Run an axis reduction (`sum_axis`/`mean_axis`/`min_axis`/`max_axis`) and
    /// wrap the 1-D result (the reduced axis is removed). Pure-Rust helper that
    /// factors the shared layout/dispatch/shape logic out of the four methods.
    fn axis_reduce(&self, py: Python<'_>, axis: usize, op: &str) -> PyResult<Self> {
        let (rows, cols) = self.require_axis_2d(op, axis)?;
        let layout =
            Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    let operand = hephaestus_wgpu::StridedOperand {
                        buffer,
                        layout: &layout,
                    };
                    match op {
                        "sum_axis" => hephaestus_wgpu::sum_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        "mean_axis" => hephaestus_wgpu::mean_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        "min_axis" => hephaestus_wgpu::min_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        "max_axis" => hephaestus_wgpu::max_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        _ => unreachable!("invariant: axis reducer is selected by wrapper"),
                    }
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    let operand = hephaestus_cuda::StridedOperand {
                        buffer,
                        layout: &layout,
                    };
                    match op {
                        "sum_axis" => hephaestus_cuda::sum_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        "mean_axis" => hephaestus_cuda::mean_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        "min_axis" => hephaestus_cuda::min_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        "max_axis" => hephaestus_cuda::max_axis(
                            device,
                            operand,
                            axis,
                            hephaestus_core::BlockWidth::DEFAULT,
                        ),
                        _ => unreachable!("invariant: axis reducer is selected by wrapper"),
                    }
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
                }
                _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "array buffer belongs to a different backend".to_string(),
                }),
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        // Reducing along `axis` removes it: axis 0 -> [cols], axis 1 -> [rows].
        let out_len = if axis == 0 { cols } else { rows };
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![out_len],
        })
    }
}

#[pymethods]
impl PyArray {
    // ── Reductions ──

    fn sum(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_reduction!(&dev, &buf, hephaestus_wgpu::SumOp, hephaestus_cuda::SumOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn min(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_reduction!(&dev, &buf, hephaestus_wgpu::MinOp, hephaestus_cuda::MinOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn max(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_reduction!(&dev, &buf, hephaestus_wgpu::MaxOp, hephaestus_cuda::MaxOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn mean(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let len = self.buffer.len();
        let out_buf = py
            .detach(move || {
                let summed =
                    backend_reduction!(&dev, &buf, hephaestus_wgpu::SumOp, hephaestus_cuda::SumOp)?;
                backend_scalar!(
                    &dev,
                    &summed,
                    1.0 / len as f32,
                    hephaestus_wgpu::MulOp,
                    hephaestus_cuda::MulOp
                )
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    /// Sum along `axis` (0 or 1); the reduced axis is removed (1-D result).
    fn sum_axis(&self, py: Python<'_>, axis: usize) -> PyResult<Self> {
        self.axis_reduce(py, axis, "sum_axis")
    }

    /// Mean along `axis` (0 or 1); the reduced axis is removed (1-D result).
    fn mean_axis(&self, py: Python<'_>, axis: usize) -> PyResult<Self> {
        self.axis_reduce(py, axis, "mean_axis")
    }

    /// Minimum along `axis` (0 or 1); the reduced axis is removed (1-D result).
    fn min_axis(&self, py: Python<'_>, axis: usize) -> PyResult<Self> {
        self.axis_reduce(py, axis, "min_axis")
    }

    /// Maximum along `axis` (0 or 1); the reduced axis is removed (1-D result).
    fn max_axis(&self, py: Python<'_>, axis: usize) -> PyResult<Self> {
        self.axis_reduce(py, axis, "max_axis")
    }

    fn norm_l1(&self, py: Python<'_>) -> PyResult<Self> {
        let out_buf = match self.shape.len() {
            1 => {
                let layout = Layout::c_contiguous([self.shape[0]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let dev = self.device.clone();
                let buf = self.buffer.clone();
                py.detach(move || {
                    backend_norm!(
                        &dev,
                        &buf,
                        layout,
                        hephaestus_wgpu::norm_l1,
                        hephaestus_cuda::norm_l1
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            2 => {
                let layout = Layout::c_contiguous([self.shape[0], self.shape[1]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let dev = self.device.clone();
                let buf = self.buffer.clone();
                py.detach(move || {
                    backend_norm!(
                        &dev,
                        &buf,
                        layout,
                        hephaestus_wgpu::norm_l1,
                        hephaestus_cuda::norm_l1
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            _ => return Err(PyValueError::new_err("norm only supports 1D or 2D arrays")),
        };
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn norm_l2(&self, py: Python<'_>) -> PyResult<Self> {
        let out_buf = match self.shape.len() {
            1 => {
                let layout = Layout::c_contiguous([self.shape[0]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let dev = self.device.clone();
                let buf = self.buffer.clone();
                py.detach(move || {
                    backend_norm!(
                        &dev,
                        &buf,
                        layout,
                        hephaestus_wgpu::norm_l2,
                        hephaestus_cuda::norm_l2
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            2 => {
                let layout = Layout::c_contiguous([self.shape[0], self.shape[1]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let dev = self.device.clone();
                let buf = self.buffer.clone();
                py.detach(move || {
                    backend_norm!(
                        &dev,
                        &buf,
                        layout,
                        hephaestus_wgpu::norm_l2,
                        hephaestus_cuda::norm_l2
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            _ => return Err(PyValueError::new_err("norm only supports 1D or 2D arrays")),
        };
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn norm_max(&self, py: Python<'_>) -> PyResult<Self> {
        let out_buf = match self.shape.len() {
            1 => {
                let layout = Layout::c_contiguous([self.shape[0]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let dev = self.device.clone();
                let buf = self.buffer.clone();
                py.detach(move || {
                    backend_norm!(
                        &dev,
                        &buf,
                        layout,
                        hephaestus_wgpu::norm_max,
                        hephaestus_cuda::norm_max
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            2 => {
                let layout = Layout::c_contiguous([self.shape[0], self.shape[1]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let dev = self.device.clone();
                let buf = self.buffer.clone();
                py.detach(move || {
                    backend_norm!(
                        &dev,
                        &buf,
                        layout,
                        hephaestus_wgpu::norm_max,
                        hephaestus_cuda::norm_max
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            _ => return Err(PyValueError::new_err("norm only supports 1D or 2D arrays")),
        };
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }
}

// ── Module-level function forms ──

#[pyfunction]
pub(crate) fn sum(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.sum(py)
}

#[pyfunction]
pub(crate) fn min(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.min(py)
}

#[pyfunction]
pub(crate) fn max(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.max(py)
}

#[pyfunction]
pub(crate) fn mean(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.mean(py)
}

#[pyfunction]
#[pyo3(name = "norm_l1")]
pub(crate) fn norm_l1_py(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.norm_l1(py)
}

#[pyfunction]
#[pyo3(name = "norm_l2")]
pub(crate) fn norm_l2_py(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.norm_l2(py)
}

#[pyfunction]
#[pyo3(name = "norm_max")]
pub(crate) fn norm_max_py(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    a.norm_max(py)
}
