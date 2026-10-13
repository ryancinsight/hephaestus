use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice, clone_cuda_buffer};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyfunction]
pub(crate) fn cholesky(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "cholesky requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();
    let lower = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                hephaestus_wgpu::cholesky_decompose_blocked(device, op)
                    .map(|decomp| BackendBuffer::Wgpu(decomp.into_lower()))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                hephaestus_cuda::cholesky_decompose_blocked(device, op)
                    .map(|decomp| BackendBuffer::Cuda(Arc::new(decomp.into_lower())))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray {
        buffer: lower,
        device: a.device.clone(),
        shape: vec![n, n],
    })
}

#[pyfunction]
pub(crate) fn bunch_kaufman(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<u64>)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "bunch_kaufman requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (l_buf, d_buf, perm, n_val) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::bunch_kaufman(device, op)?;
                let perm = decomp.permutation().iter().map(|&x| x as u64).collect();
                Ok((
                    BackendBuffer::Wgpu(decomp.l_buffer().clone()),
                    BackendBuffer::Wgpu(decomp.d_buffer().clone()),
                    perm,
                    decomp.n(),
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::bunch_kaufman(device, op)?;
                let perm = decomp.permutation().iter().map(|&x| x as u64).collect();
                Ok((
                    clone_cuda_buffer(device, decomp.l_buffer())?,
                    clone_cuda_buffer(device, decomp.d_buffer())?,
                    perm,
                    decomp.n(),
                ))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
