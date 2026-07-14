//! Seeded random-number initialisers (uniform, normal).

use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice};
use crate::device::PyDevice;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyfunction]
pub(crate) fn uniform_with_seed(
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
        .detach(move || match (&dev, shape_cloned.as_slice()) {
            (BackendDevice::Wgpu(device), [n]) => {
                hephaestus_wgpu::uniform_with_seed(device, [*n], low, high, seed)
                    .map(BackendBuffer::Wgpu)
            }
            (BackendDevice::Wgpu(device), [rows, cols]) => {
                hephaestus_wgpu::uniform_with_seed(device, [*rows, *cols], low, high, seed)
                    .map(BackendBuffer::Wgpu)
            }
            (BackendDevice::Cuda(device), [n]) => {
                hephaestus_cuda::uniform_with_seed(device, [*n], low, high, seed)
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            (BackendDevice::Cuda(device), [rows, cols]) => {
                hephaestus_cuda::uniform_with_seed(device, [*rows, *cols], low, high, seed)
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            (_, shape) => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: format!(
                    "RNG only supports 1D or 2D shapes, got rank {}",
                    shape.len()
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
pub(crate) fn normal_with_seed(
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
        .detach(move || match (&dev, shape_cloned.as_slice()) {
            (BackendDevice::Wgpu(device), [n]) => {
                hephaestus_wgpu::normal_with_seed(device, [*n], mean, std_dev, seed)
                    .map(BackendBuffer::Wgpu)
            }
            (BackendDevice::Wgpu(device), [rows, cols]) => {
                hephaestus_wgpu::normal_with_seed(device, [*rows, *cols], mean, std_dev, seed)
                    .map(BackendBuffer::Wgpu)
            }
            (BackendDevice::Cuda(device), [n]) => {
                hephaestus_cuda::normal_with_seed(device, [*n], mean, std_dev, seed)
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            (BackendDevice::Cuda(device), [rows, cols]) => {
                hephaestus_cuda::normal_with_seed(device, [*rows, *cols], mean, std_dev, seed)
                    .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            (_, shape) => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: format!(
                    "RNG only supports 1D or 2D shapes, got rank {}",
                    shape.len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::prepare_python;

    #[test]
    fn test_py_rng_initializers() {
        prepare_python();
        Python::attach(|py| {
            let device = PyDevice::new(None).unwrap();
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
