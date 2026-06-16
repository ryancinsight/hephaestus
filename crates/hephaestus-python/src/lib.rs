// ── Hephaestus Python Bindings (pyhephaestus) ──

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use hephaestus_wgpu::{
    WgpuDevice, WgpuBuffer,
    AddOp, SubOp, MulOp,
    ExpOp, LnOp, SinOp, CosOp, SqrtOp, AbsOp, NegOp,
    SumOp, MinOp, MaxOp,
    ComputeDevice, DeviceBuffer,
};


/// Python wrapper around WgpuDevice.
#[pyclass(name = "Device")]
#[derive(Clone)]
pub struct PyDevice {
    pub inner: WgpuDevice,
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
}

#[pymethods]
impl PyArray {
    /// Upload a python list/iterable of floats to the GPU.
    #[new]
    #[pyo3(signature = (data, device))]
    fn new(data: Vec<f32>, device: &PyDevice) -> PyResult<Self> {
        let buffer = device.inner.upload(&data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
        })
    }

    /// Allocate a zeroed array of a given length on the GPU.
    #[staticmethod]
    fn zeros(len: usize, device: &PyDevice) -> PyResult<Self> {
        let buffer = device.inner.alloc_zeroed::<f32>(len)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
        })
    }

    /// Create an Array from a contiguous NumPy array.
    #[staticmethod]
    fn from_numpy(arr: PyReadonlyArray1<'_, f32>, device: &PyDevice) -> PyResult<Self> {
        let slice = arr.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let buffer = device.inner.upload(slice)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer,
            device: device.inner.clone(),
        })
    }

    /// Download array data to a Python list.
    fn tolist(&self) -> PyResult<Vec<f32>> {
        let mut host_data = vec![0.0f32; self.buffer.len()];
        self.device.download(&self.buffer, &mut host_data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(host_data)
    }

    /// Download array data to a NumPy 1D array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mut host_data = vec![0.0f32; self.buffer.len()];
        self.device.download(&self.buffer, &mut host_data)
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
        })
    }

    fn log(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<LnOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn sin(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<SinOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn cos(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<CosOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn sqrt(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<SqrtOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn abs(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<AbsOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn neg(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::unary_elementwise::<NegOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    // ── Reductions ──

    fn sum(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::reduction::<SumOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn min(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::reduction::<MinOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    fn max(&self) -> PyResult<Self> {
        let out_buf = hephaestus_wgpu::reduction::<MaxOp, f32>(&self.device, &self.buffer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
        })
    }

    // ── Binary Operations ──

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            let out_buf = hephaestus_wgpu::binary_elementwise::<AddOp, f32>(&self.device, &self.buffer, &other_arr.buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf = hephaestus_wgpu::scalar_elementwise::<AddOp, f32>(&self.device, &self.buffer, val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
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
            let out_buf = hephaestus_wgpu::binary_elementwise::<SubOp, f32>(&self.device, &self.buffer, &other_arr.buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf = hephaestus_wgpu::scalar_elementwise::<SubOp, f32>(&self.device, &self.buffer, val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for -"))
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f32>() {
            let negated = hephaestus_wgpu::unary_elementwise::<NegOp, f32>(&self.device, &self.buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let out_buf = hephaestus_wgpu::scalar_elementwise::<AddOp, f32>(&self.device, &negated, val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for -"))
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyArray>>() {
            let out_buf = hephaestus_wgpu::binary_elementwise::<MulOp, f32>(&self.device, &self.buffer, &other_arr.buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
            })
        } else if let Ok(val) = other.extract::<f32>() {
            let out_buf = hephaestus_wgpu::scalar_elementwise::<MulOp, f32>(&self.device, &self.buffer, val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                buffer: out_buf,
                device: self.device.clone(),
            })
        } else {
            Err(PyTypeError::new_err("unsupported operand type(s) for *"))
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
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

/// PyHephaestus extension module definition.
#[pymodule]
fn pyhephaestus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyArray>()?;

    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;

    Ok(())
}
