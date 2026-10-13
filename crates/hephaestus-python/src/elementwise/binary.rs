use crate::array::PyArray;
use crate::backend::{backend_binary, backend_scalar, backend_unary};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pymethods]
impl PyArray {
    pub(crate) fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
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
                .detach(move || {
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
                .detach(move || {
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

    pub(crate) fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }

    pub(crate) fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
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
                .detach(move || {
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
                .detach(move || {
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

    pub(crate) fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .detach(move || {
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

    pub(crate) fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
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
                .detach(move || {
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
                .detach(move || {
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

    pub(crate) fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }

    pub(crate) fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
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
                .detach(move || {
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
                .detach(move || {
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

    pub(crate) fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let dev = self.device.clone();
            let buf = self.buffer.clone();
            let out_buf = py
                .detach(move || {
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

    pub(crate) fn __pow__(
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
                .detach(move || {
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
                .detach(move || {
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

    pub(crate) fn __rpow__(
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
                .detach(move || {
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
