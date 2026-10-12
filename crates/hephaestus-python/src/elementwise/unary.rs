use crate::array::PyArray;
use crate::backend::backend_unary;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pymethods]
impl PyArray {
    // ── Unary Operations ──

    pub(crate) fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::ExpOp, hephaestus_cuda::ExpOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    pub(crate) fn log(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::LnOp, hephaestus_cuda::LnOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    pub(crate) fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::SinOp, hephaestus_cuda::SinOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    pub(crate) fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::CosOp, hephaestus_cuda::CosOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    pub(crate) fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::SqrtOp, hephaestus_cuda::SqrtOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    pub(crate) fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::AbsOp, hephaestus_cuda::AbsOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }

    pub(crate) fn neg(&self, py: Python<'_>) -> PyResult<Self> {
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .detach(move || {
                backend_unary!(&dev, &buf, hephaestus_wgpu::NegOp, hephaestus_cuda::NegOp)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            buffer: out_buf,
            device: self.device.clone(),
            shape: self.shape.clone(),
        })
    }
}
