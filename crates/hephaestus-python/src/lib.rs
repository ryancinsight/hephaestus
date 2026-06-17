// ── Hephaestus Python Bindings (pyhephaestus) ──

use hephaestus_wgpu::{
    dot, matmul, norm_l1, norm_l2, norm_max, trace, AbsOp, AddOp, ComputeDevice, CosOp,
    DeviceBuffer, DivOp, ExpOp, LnOp, MaxOp, MinOp, MulOp, NegOp, PowOp, RecipOp, SinOp, SqrtOp,
    StridedOperand, SubOp, SumOp, WgpuBuffer, WgpuDevice,
};
use leto::Layout;
use num_complex::Complex;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Python wrapper around WgpuDevice.
#[pyclass(name = "Device")]
#[derive(Clone)]
pub struct PyDevice {
    pub inner: WgpuDevice,
}

impl Drop for PyDevice {
    fn drop(&mut self) {
        self.inner.clear_transient_pools();
    }
}

#[pymethods]
impl PyDevice {
    /// Create a new device context.
    #[new]
    fn new() -> PyResult<Self> {
        let device = WgpuDevice::try_default("hephaestus-py-device")
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: device })
    }

    /// Get the backend name.
    #[getter]
    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }
}

/// Python wrapper around a GPU-resident WgpuBuffer<f32>.
#[pyclass(name = "Array")]
pub struct PyArray {
    pub buffer: WgpuBuffer<f32>,
    pub device: WgpuDevice,
    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl PyArray {
    /// Upload a python list/iterable of floats to the GPU.
    #[new]
    #[pyo3(signature = (data, device))]
    fn new(py: Python<'_>, data: Vec<f32>, device: &PyDevice) -> PyResult<Self> {
        let len = data.len();
        let dev = device.inner.clone();
        let buffer = py
            .allow_threads(move || dev.upload(&data))
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
            .alloc_zeroed::<f32>(len)
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
            .allow_threads(move || dev.upload(slice))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
            shape: vec![len],
        })
    }

    /// Reshape the array to a new shape.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
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
    fn tolist(&self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let len = self.buffer.len();
        let host_data = py
            .allow_threads(move || {
                let mut host_data = vec![0.0f32; len];
                dev.download(&buf, &mut host_data).map(|_| host_data)
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
                dev.download(&buf, &mut host_data).map(|_| host_data)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(host_data.to_pyarray(py))
    }

    /// Get the length of the array.
    #[getter]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    // ── Unary Operations ──

    fn exp(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<ExpOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn log(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<LnOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn sin(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<SinOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn cos(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<CosOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn sqrt(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<SqrtOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn abs(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<AbsOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    fn neg(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<NegOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    // ── Reductions ──

    fn sum(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::reduction::<SumOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn min(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::reduction::<MinOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn max(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::reduction::<MaxOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn mean(&self) -> PyResult<Self> {
        let summed = hephaestus_wgpu::reduction::<SumOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out_buf = hephaestus_wgpu::scalar_elementwise::<MulOp, f32>(
            &self.device,
            &summed,
            1.0 / self.buffer.len() as f32,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    // ── Binary Operations ──

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let out_buf = hephaestus_wgpu::binary_elementwise::<AddOp, f32>(
                &self.device,
                &self.buffer,
                &other_arr.buffer,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<AddOp, f32>(&self.device, &self.buffer, val)
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

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let out_buf = hephaestus_wgpu::binary_elementwise::<SubOp, f32>(
                &self.device,
                &self.buffer,
                &other_arr.buffer,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<SubOp, f32>(&self.device, &self.buffer, val)
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

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let negated =
                hephaestus_wgpu::unary_elementwise::<NegOp, f32>(&self.device, &self.buffer)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<AddOp, f32>(&self.device, &negated, val)
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

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let out_buf = hephaestus_wgpu::binary_elementwise::<MulOp, f32>(
                &self.device,
                &self.buffer,
                &other_arr.buffer,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<MulOp, f32>(&self.device, &self.buffer, val)
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

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            if self.shape != other_arr.shape {
                return Err(PyValueError::new_err(format!(
                    "shape mismatch: {:?} vs {:?}",
                    self.shape, other_arr.shape
                )));
            }
            let out_buf = hephaestus_wgpu::binary_elementwise::<DivOp, f32>(
                &self.device,
                &self.buffer,
                &other_arr.buffer,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<DivOp, f32>(&self.device, &self.buffer, val)
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

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let recip =
                hephaestus_wgpu::unary_elementwise::<RecipOp, f32>(&self.device, &self.buffer)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<MulOp, f32>(&self.device, &recip, val)
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
            let out_buf = hephaestus_wgpu::binary_elementwise::<PowOp, f32>(
                &self.device,
                &self.buffer,
                &other_arr.buffer,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
                shape: self.shape.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf =
                hephaestus_wgpu::scalar_elementwise::<PowOp, f32>(&self.device, &self.buffer, val)
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
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            if val <= 0.0 {
                return Err(PyValueError::new_err("power base must be positive"));
            }
            let ln_val = val.ln();
            let scaled = hephaestus_wgpu::scalar_elementwise::<MulOp, f32>(
                &self.device,
                &self.buffer,
                ln_val,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let out_buf = hephaestus_wgpu::unary_elementwise::<ExpOp, f32>(&self.device, &scaled)
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

    fn matmul(&self, other: &PyArray) -> PyResult<Self> {
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
        let out_buf = matmul(
            &self.device,
            StridedOperand {
                buffer: &self.buffer,
                layout: &layout_a,
            },
            StridedOperand {
                buffer: &other.buffer,
                layout: &layout_b,
            },
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![m, n],
        })
    }

    fn dot(&self, other: &PyArray) -> PyResult<Self> {
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

        let out_buf = dot(
            &self.device,
            StridedOperand {
                buffer: &self.buffer,
                layout: &layout_a,
            },
            StridedOperand {
                buffer: &other.buffer,
                layout: &layout_b,
            },
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn trace(&self) -> PyResult<Self> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err("trace requires a 2D array"));
        }
        if self.shape[0] != self.shape[1] {
            return Err(PyValueError::new_err("trace requires a square matrix"));
        }
        let n = self.shape[0];
        let layout =
            Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let out_buf = trace(
            &self.device,
            StridedOperand {
                buffer: &self.buffer,
                layout: &layout,
            },
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: vec![1],
        })
    }

    fn norm_l1(&self) -> PyResult<Self> {
        let out_buf = match self.shape.len() {
            1 => {
                let layout = Layout::c_contiguous([self.shape[0]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                norm_l1(
                    &self.device,
                    StridedOperand {
                        buffer: &self.buffer,
                        layout: &layout,
                    },
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            2 => {
                let layout = Layout::c_contiguous([self.shape[0], self.shape[1]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                norm_l1(
                    &self.device,
                    StridedOperand {
                        buffer: &self.buffer,
                        layout: &layout,
                    },
                )
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

    fn norm_l2(&self) -> PyResult<Self> {
        let out_buf = match self.shape.len() {
            1 => {
                let layout = Layout::c_contiguous([self.shape[0]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                norm_l2(
                    &self.device,
                    StridedOperand {
                        buffer: &self.buffer,
                        layout: &layout,
                    },
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            2 => {
                let layout = Layout::c_contiguous([self.shape[0], self.shape[1]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                norm_l2(
                    &self.device,
                    StridedOperand {
                        buffer: &self.buffer,
                        layout: &layout,
                    },
                )
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

    fn norm_max(&self) -> PyResult<Self> {
        let out_buf = match self.shape.len() {
            1 => {
                let layout = Layout::c_contiguous([self.shape[0]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                norm_max(
                    &self.device,
                    StridedOperand {
                        buffer: &self.buffer,
                        layout: &layout,
                    },
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            2 => {
                let layout = Layout::c_contiguous([self.shape[0], self.shape[1]])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                norm_max(
                    &self.device,
                    StridedOperand {
                        buffer: &self.buffer,
                        layout: &layout,
                    },
                )
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

// ── Top-level functions ──

#[pyfunction]
fn add(a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__add__(b)
}

#[pyfunction]
fn sub(a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__sub__(b)
}

#[pyfunction]
fn mul(a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__mul__(b)
}

#[pyfunction]
fn div(a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__truediv__(b)
}

#[pyfunction]
fn pow(a: &PyArray, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    a.__pow__(b, None)
}

#[pyfunction]
fn exp(a: &PyArray) -> PyResult<PyArray> {
    a.exp()
}

#[pyfunction]
fn log(a: &PyArray) -> PyResult<PyArray> {
    a.log()
}

#[pyfunction]
fn sin(a: &PyArray) -> PyResult<PyArray> {
    a.sin()
}

#[pyfunction]
fn cos(a: &PyArray) -> PyResult<PyArray> {
    a.cos()
}

#[pyfunction]
fn sqrt(a: &PyArray) -> PyResult<PyArray> {
    a.sqrt()
}

#[pyfunction]
fn abs(a: &PyArray) -> PyResult<PyArray> {
    a.abs()
}

#[pyfunction]
fn neg(a: &PyArray) -> PyResult<PyArray> {
    a.neg()
}

#[pyfunction]
fn sum(a: &PyArray) -> PyResult<PyArray> {
    a.sum()
}

#[pyfunction]
fn min(a: &PyArray) -> PyResult<PyArray> {
    a.min()
}

#[pyfunction]
fn max(a: &PyArray) -> PyResult<PyArray> {
    a.max()
}

#[pyfunction]
fn mean(a: &PyArray) -> PyResult<PyArray> {
    a.mean()
}

#[pyfunction]
#[pyo3(name = "matmul")]
fn matmul_py(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    a.matmul(b)
}

#[pyfunction]
#[pyo3(name = "dot")]
fn dot_py(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    a.dot(b)
}

#[pyfunction]
#[pyo3(name = "trace")]
fn trace_py(a: &PyArray) -> PyResult<PyArray> {
    a.trace()
}

#[pyfunction]
#[pyo3(name = "norm_l1")]
fn norm_l1_py(a: &PyArray) -> PyResult<PyArray> {
    a.norm_l1()
}

#[pyfunction]
#[pyo3(name = "norm_l2")]
fn norm_l2_py(a: &PyArray) -> PyResult<PyArray> {
    a.norm_l2()
}

#[pyfunction]
#[pyo3(name = "norm_max")]
fn norm_max_py(a: &PyArray) -> PyResult<PyArray> {
    a.norm_max()
}

#[pyfunction]
fn cholesky(a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "cholesky requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::cholesky_decompose_blocked(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray {
        buffer: decomp.into_lower(),
        device: a.device.clone(),
        shape: vec![n, n],
    })
}

#[pyfunction]
fn lu(a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<usize>)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("lu requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::lu_decompose_blocked(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut host_factors = vec![0.0f32; n * n];
    a.device
        .download(decomp.factors(), &mut host_factors)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut host_l = vec![0.0f32; n * n];
    let mut host_u = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let idx = r * n + c;
            let val = host_factors[idx];
            if r > c {
                host_l[idx] = val;
            } else if r == c {
                host_l[idx] = 1.0;
                host_u[idx] = val;
            } else {
                host_u[idx] = val;
            }
        }
    }

    let l_buf = a
        .device
        .upload(&host_l)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let u_buf = a
        .device
        .upload(&host_u)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: l_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
        PyArray {
            buffer: u_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
        decomp.pivots().to_vec(),
    ))
}

#[pyfunction]
fn qr(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("qr requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::qr_decompose_blocked(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let q_host = decomp.inner().q();
    let r_host = decomp.inner().r();

    let q_buf = a
        .device
        .upload(leto::Storage::as_slice(q_host.storage()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let r_buf = a
        .device
        .upload(leto::Storage::as_slice(r_host.storage()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: q_buf,
            device: a.device.clone(),
            shape: vec![q_host.shape()[0], q_host.shape()[1]],
        },
        PyArray {
            buffer: r_buf,
            device: a.device.clone(),
            shape: vec![r_host.shape()[0], r_host.shape()[1]],
        },
    ))
}

#[pyfunction]
fn col_piv_qr(a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<u64>)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("col_piv_qr requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::col_piv_qr(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let q_buf = decomp.q().clone();
    let r_buf = decomp.r().clone();
    let m = (q_buf.len() as f64).sqrt() as usize;
    let n = r_buf.len().checked_div(m).unwrap_or(0);
    let perm = decomp
        .permutation()
        .iter()
        .map(|&x| x as u64)
        .collect::<Vec<_>>();

    Ok((
        PyArray {
            buffer: q_buf,
            device: a.device.clone(),
            shape: vec![m, m],
        },
        PyArray {
            buffer: r_buf,
            device: a.device.clone(),
            shape: vec![m, n],
        },
        perm,
    ))
}

#[pyfunction]
fn svd(a: &PyArray) -> PyResult<(PyArray, PyArray, PyArray)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("svd requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::svd_decompose(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_host = decomp.inner().left_singular_vectors.clone();
    let s_host = leto::Array1::from_shape_vec(
        [decomp.inner().singular_values.len()],
        decomp.inner().singular_values.clone(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let vt_transposed = decomp
        .inner()
        .right_singular_vectors
        .transpose([1, 0])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vt_host = vt_transposed.to_contiguous();

    let u_buf = a
        .device
        .upload(leto::Storage::as_slice(u_host.storage()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let s_buf = a
        .device
        .upload(leto::Storage::as_slice(s_host.storage()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vt_buf = a
        .device
        .upload(leto::Storage::as_slice(vt_host.storage()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: u_buf,
            device: a.device.clone(),
            shape: vec![u_host.shape()[0], u_host.shape()[1]],
        },
        PyArray {
            buffer: s_buf,
            device: a.device.clone(),
            shape: vec![s_host.shape()[0]],
        },
        PyArray {
            buffer: vt_buf,
            device: a.device.clone(),
            shape: vec![vt_host.shape()[0], vt_host.shape()[1]],
        },
    ))
}

#[pyfunction]
fn symmetric_eigen(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "symmetric_eigen requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::symmetric_eigen_jacobi(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let w_host = &decomp.inner().eigenvalues;
    let v_host = decomp.inner().eigenvectors.clone();

    let w_buf = a
        .device
        .upload(w_host)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let v_buf = a
        .device
        .upload(leto::Storage::as_slice(v_host.storage()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: w_buf,
            device: a.device.clone(),
            shape: vec![w_host.len()],
        },
        PyArray {
            buffer: v_buf,
            device: a.device.clone(),
            shape: vec![v_host.shape()[0], v_host.shape()[1]],
        },
    ))
}

#[pyfunction]
fn singular_values(a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err(
            "singular_values requires a 2D matrix",
        ));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let s_buf = hephaestus_wgpu::singular_values(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: s_buf,
        device: a.device.clone(),
        shape: vec![rows.min(cols)],
    })
}

#[pyfunction]
fn schur(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("schur requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::schur(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let q_buf = decomp.q_buffer().clone();
    let t_buf = decomp.t_buffer().clone();
    let n_val = decomp.n();

    Ok((
        PyArray {
            buffer: q_buf,
            device: a.device.clone(),
            shape: vec![n_val, n_val],
        },
        PyArray {
            buffer: t_buf,
            device: a.device.clone(),
            shape: vec![n_val, n_val],
        },
    ))
}

#[pyfunction]
fn bunch_kaufman(a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<u64>)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "bunch_kaufman requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let decomp = hephaestus_wgpu::bunch_kaufman(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let l_buf = decomp.l_buffer().clone();
    let d_buf = decomp.d_buffer().clone();
    let perm = decomp
        .permutation()
        .iter()
        .map(|&x| x as u64)
        .collect::<Vec<_>>();
    let n_val = decomp.n();

    Ok((
        PyArray {
            buffer: l_buf,
            device: a.device.clone(),
            shape: vec![n_val, n_val],
        },
        PyArray {
            buffer: d_buf,
            device: a.device.clone(),
            shape: vec![n_val, n_val],
        },
        perm,
    ))
}

#[pyfunction]
fn matexp(a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("matexp requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let out_buf = hephaestus_wgpu::matexp(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![n, n],
    })
}

#[pyfunction]
fn pinv(a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("pinv requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let out_buf =
        hephaestus_wgpu::pinv(&a.device, op).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![cols, rows],
    })
}

#[pyfunction]
fn eigenvalues<'py>(
    py: Python<'py>,
    a: &PyArray,
) -> PyResult<Bound<'py, PyArray1<numpy::Complex32>>> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "eigenvalues requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let op = StridedOperand {
        buffer: &a.buffer,
        layout: &layout,
    };
    let e_buf = hephaestus_wgpu::eigenvalues(&a.device, op)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut host_data = vec![Complex::new(0.0f32, 0.0f32); n];
    a.device
        .download(&e_buf, &mut host_data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let py_data = host_data
        .into_iter()
        .map(|c| numpy::Complex32::new(c.re, c.im))
        .collect::<Vec<_>>();

    Ok(PyArray1::from_vec(py, py_data))
}

/// Python wrapper around a GPU-resident GpuCsrMatrix<f32>.
#[pyclass(name = "SparseMatrix")]
#[derive(Debug)]
pub struct PyCsrMatrix {
    pub inner: hephaestus_wgpu::GpuCsrMatrix<f32>,
    pub device: WgpuDevice,
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
                device.download(&buffer, &mut host_data).map(|_| host_data)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let view = leto::ArrayView2::new(layout, &host_data);
        let cpu_csr = leto_ops::CsrMatrix::from_dense(&view);

        let inner = hephaestus_wgpu::GpuCsrMatrix::from_cpu(&arr.device, &cpu_csr)
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
            .allow_threads(move || inner.to_cpu(&device))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let cpu_dense = cpu_csr.to_dense();
        let dense_buf = self
            .device
            .upload(leto::Storage::as_slice(cpu_dense.storage()))
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
fn spmv(py: Python<'_>, a: &PyCsrMatrix, x: &PyArray) -> PyResult<PyArray> {
    let device = a.device.clone();
    let inner_a = a.inner.clone();
    let buf_x = x.buffer.clone();

    let out_buf = py
        .allow_threads(move || hephaestus_wgpu::spmv(&device, &inner_a, &buf_x))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let (rows, _) = a.inner.shape();
    Ok(PyArray {
        buffer: out_buf,
        device: a.device.clone(),
        shape: vec![rows],
    })
}

#[pyfunction]
fn spmm(py: Python<'_>, a: &PyCsrMatrix, b: &PyArray) -> PyResult<PyArray> {
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
        .allow_threads(move || {
            let op_b = StridedOperand {
                buffer: &buf_b,
                layout: &layout_b,
            };
            hephaestus_wgpu::spmm(&device, &inner_a, &op_b)
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
fn uniform_with_seed(
    py: Python<'_>,
    shape: Vec<usize>,
    low: f32,
    high: f32,
    seed: u64,
    device: &PyDevice,
) -> PyResult<PyArray> {
    let dev = device.inner.clone();
    let shape_cloned = shape.clone();
    let out_buf = py
        .allow_threads(move || match shape_cloned.len() {
            1 => hephaestus_wgpu::uniform_with_seed(&dev, [shape_cloned[0]], low, high, seed),
            2 => hephaestus_wgpu::uniform_with_seed(
                &dev,
                [shape_cloned[0], shape_cloned[1]],
                low,
                high,
                seed,
            ),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: format!(
                    "RNG only supports 1D or 2D shapes, got rank {}",
                    shape_cloned.len()
                ),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: out_buf,
        device: device.inner.clone(),
        shape,
    })
}

#[pyfunction]
fn normal_with_seed(
    py: Python<'_>,
    shape: Vec<usize>,
    mean: f32,
    std_dev: f32,
    seed: u64,
    device: &PyDevice,
) -> PyResult<PyArray> {
    let dev = device.inner.clone();
    let shape_cloned = shape.clone();
    let out_buf = py
        .allow_threads(move || match shape_cloned.len() {
            1 => hephaestus_wgpu::normal_with_seed(&dev, [shape_cloned[0]], mean, std_dev, seed),
            2 => hephaestus_wgpu::normal_with_seed(
                &dev,
                [shape_cloned[0], shape_cloned[1]],
                mean,
                std_dev,
                seed,
            ),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: format!(
                    "RNG only supports 1D or 2D shapes, got rank {}",
                    shape_cloned.len()
                ),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: out_buf,
        device: device.inner.clone(),
        shape,
    })
}

/// PyHephaestus extension module definition.
#[pymodule]
fn pyhephaestus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyCsrMatrix>()?;

    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(dot_py, m)?)?;
    m.add_function(wrap_pyfunction!(trace_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_l1_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_l2_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_max_py, m)?)?;

    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(lu, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(col_piv_qr, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_eigen, m)?)?;
    m.add_function(wrap_pyfunction!(singular_values, m)?)?;
    m.add_function(wrap_pyfunction!(schur, m)?)?;
    m.add_function(wrap_pyfunction!(bunch_kaufman, m)?)?;
    m.add_function(wrap_pyfunction!(matexp, m)?)?;
    m.add_function(wrap_pyfunction!(pinv, m)?)?;
    m.add_function(wrap_pyfunction!(eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(spmv, m)?)?;
    m.add_function(wrap_pyfunction!(spmm, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_with_seed, m)?)?;
    m.add_function(wrap_pyfunction!(normal_with_seed, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArrayMethods;
    use std::sync::Once;

    static INIT_PYTHON: Once = Once::new();

    fn prepare_python() {
        INIT_PYTHON.call_once(pyo3::prepare_freethreaded_python);
    }

    #[test]
    fn test_py_array_tolist_and_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let device = PyDevice::new().unwrap();
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let py_arr = PyArray::new(py, data.clone(), &device).unwrap();
            assert_eq!(py_arr.tolist(py).unwrap(), data);

            let np_arr = py_arr.to_numpy(py).unwrap();
            assert_eq!(np_arr.readonly().as_slice().unwrap(), &data[..]);
        });
    }

    #[test]
    fn test_py_sparse_matrix_roundtrip_spmv_spmm() {
        prepare_python();
        Python::with_gil(|py| {
            let device = PyDevice::new().unwrap();

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
        });
    }

    #[test]
    fn test_py_rng_initializers() {
        prepare_python();
        Python::with_gil(|py| {
            let device = PyDevice::new().unwrap();
            let u = uniform_with_seed(py, vec![100], -1.0, 2.0, 13, &device).unwrap();
            assert_eq!(u.shape, vec![100]);
            let u_list = u.tolist(py).unwrap();
            for &val in &u_list {
                assert!((-1.0..2.0).contains(&val));
            }

            let n = normal_with_seed(py, vec![100], 0.0, 1.0, 13, &device).unwrap();
            assert_eq!(n.shape, vec![100]);
            let n_list = n.tolist(py).unwrap();
            assert!(n_list.iter().any(|&val| val != 0.0));
        });
    }
}
