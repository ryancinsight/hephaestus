//! Prefix scans on `Array` (cumulative sum along an axis).

use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyArray {
    /// Cumulative sum along `axis` (0 or 1); output keeps the input shape.
    fn cumsum(&self, py: Python<'_>, axis: usize) -> PyResult<Self> {
        let (rows, cols) = self.require_axis_2d("cumsum", axis)?;
        let layout =
            Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = self.device.clone();
        let buf = self.buffer.clone();
        let out_buf = py
            .allow_threads(move || match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    hephaestus_wgpu::cumsum(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                        axis,
                        hephaestus_core::BlockWidth::DEFAULT,
                    )
                    .map(BackendBuffer::Wgpu)
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    hephaestus_cuda::cumsum(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                        axis,
                        hephaestus_core::BlockWidth::DEFAULT,
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
            shape: vec![rows, cols],
        })
    }
}
