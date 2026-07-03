//! Elementwise operations on `Array`: unary transcendentals, binary
//! arithmetic dunders (array-array and array-scalar), and their
//! module-level function forms.

use crate::array::PyArray;
use crate::backend::{backend_binary, backend_scalar, backend_unary};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pymethods]
impl PyArray {
    // ── Unary Operations ──

    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::ExpOp, hephaestus_cuda::ExpOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::LnOp, hephaestus_cuda::LnOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::SinOp, hephaestus_cuda::SinOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::CosOp, hephaestus_cuda::CosOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::SqrtOp, hephaestus_cuda::SqrtOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::AbsOp, hephaestus_cuda::AbsOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn neg(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::NegOp, hephaestus_cuda::NegOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    // ── Binary Operations ──

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let other_buf = other_arr.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_binary!(
                        &dev,
                        &buf,
                        &other_buf,
                        hephaestus_wgpu::AddOp,
                        hephaestus_cuda::AddOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_scalar!(
                        &dev,
                        &buf,
                        val,
                        hephaestus_wgpu::AddOp,
                        hephaestus_cuda::AddOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for +"))
        }
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let other_buf = other_arr.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_binary!(
                        &dev,
                        &buf,
                        &other_buf,
                        hephaestus_wgpu::SubOp,
                        hephaestus_cuda::SubOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_scalar!(
                        &dev,
                        &buf,
                        val,
                        hephaestus_wgpu::SubOp,
                        hephaestus_cuda::SubOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for -"))
        }
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    let negated =
                        backend_unary!(&dev, &buf, hephaestus_wgpu::NegOp, hephaestus_cuda::NegOp)?;
                    backend_scalar!(
                        &dev,
                        &negated,
                        val,
                        hephaestus_wgpu::AddOp,
                        hephaestus_cuda::AddOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for -"))
        }
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let other_buf = other_arr.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_binary!(
                        &dev,
                        &buf,
                        &other_buf,
                        hephaestus_wgpu::MulOp,
                        hephaestus_cuda::MulOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_scalar!(
                        &dev,
                        &buf,
                        val,
                        hephaestus_wgpu::MulOp,
                        hephaestus_cuda::MulOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for *"))
        }
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let other_buf = other_arr.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_binary!(
                        &dev,
                        &buf,
                        &other_buf,
                        hephaestus_wgpu::DivOp,
                        hephaestus_cuda::DivOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_scalar!(
                        &dev,
                        &buf,
                        val,
                        hephaestus_wgpu::DivOp,
                        hephaestus_cuda::DivOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for /"))
        }
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    let recip = backend_unary!(
                        &dev,
                        &buf,
                        hephaestus_wgpu::RecipOp,
                        hephaestus_cuda::RecipOp
                    )?;
                    backend_scalar!(
                        &dev,
                        &recip,
                        val,
                        hephaestus_wgpu::MulOp,
                        hephaestus_cuda::MulOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for /"))
        }
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let other_buf = other_arr.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_binary!(
                        &dev,
                        &buf,
                        &other_buf,
                        hephaestus_wgpu::PowOp,
                        hephaestus_cuda::PowOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .allow_threads(move || {
                    backend_scalar!(
                        &dev,
                        &buf,
                        val,
                        hephaestus_wgpu::PowOp,
                        hephaestus_cuda::PowOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for **"))
        }
    }

    fn __rpow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            if val <= 0.0 {
                return Err(PyValueError::new_err("power base must be positive"));
            }
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let ln_val = val.ln();
            let out_buf = py
                .allow_threads(move || {
                    let scaled = backend_scalar!(
                        &dev,
                        &buf,
                        ln_val,
                        hephaestus_wgpu::MulOp,
                        hephaestus_cuda::MulOp
                    )?;
                    backend_unary!(
                        &dev,
                        &scaled,
                        hephaestus_wgpu::ExpOp,
                        hephaestus_cuda::ExpOp
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for **"))
        }
    }
}

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
