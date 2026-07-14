//! Python-visible `Device` class: backend selection and lifecycle.

use crate::backend::BackendDevice;
use pyo3::prelude::*;

/// Python wrapper around a compute device.
#[pyclass(name = "Device", from_py_object)]
#[derive(Clone)]
pub struct PyDevice {
    pub(crate) inner: BackendDevice,
}

impl Drop for PyDevice {
    fn drop(&mut self) {
        if let BackendDevice::Wgpu(device) = &self.inner {
            device.clear_transient_pools();
        }
    }
}

#[pymethods]
impl PyDevice {
    /// Create a new device context.
    #[new]
    #[pyo3(signature = (backend = None))]
    pub(crate) fn new(backend: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            inner: BackendDevice::try_default(backend)?,
        })
    }

    /// Get the backend name.
    #[getter]
    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }
}
