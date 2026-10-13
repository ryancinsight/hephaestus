use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyfunction]
pub(crate) fn lu(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<usize>)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("lu requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (decomp, l_buf, u_buf) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::lu_decompose_blocked(device, op)?;
                let (l, u) = hephaestus_wgpu::split_packed_lu(device, decomp.factors(), n)?;
                Ok((
                    decomp.pivots().to_vec(),
                    BackendBuffer::Wgpu(l),
                    BackendBuffer::Wgpu(u),
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::lu_decompose_blocked(device, op)?;
                let (lower, upper) = hephaestus_cuda::split_packed_lu(device, decomp.factors(), n)?;
                let l_buf = BackendBuffer::Cuda(Arc::new(lower));
                let u_buf = BackendBuffer::Cuda(Arc::new(upper));
                Ok((decomp.pivots().to_vec(), l_buf, u_buf))
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
            shape: vec![n, n],
        },
        PyArray {
            buffer: u_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
        decomp,
    ))
}

/// Full-pivoting LU on the GPU: returns `(L, U, row_perm, col_perm)` with
/// `A[row_perm][:, col_perm] = L U`, `L` unit-lower, `U` upper.
#[pyfunction]
pub(crate) fn full_piv_lu(
    py: Python<'_>,
    a: &PyArray,
) -> PyResult<(PyArray, PyArray, Vec<usize>, Vec<usize>)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "full_piv_lu requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (l_buf, u_buf, row_perm, col_perm) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::full_piv_lu(device, op)?;
                let (l, u) = hephaestus_wgpu::split_packed_lu(device, decomp.lu_buffer(), n)?;
                Ok((
                    BackendBuffer::Wgpu(l),
                    BackendBuffer::Wgpu(u),
                    decomp.row_permutation().to_vec(),
                    decomp.col_permutation().to_vec(),
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::full_piv_lu(device, op)?;
                let (lower, upper) =
                    hephaestus_cuda::split_packed_lu(device, decomp.lu_buffer(), n)?;
                Ok((
                    BackendBuffer::Cuda(Arc::new(lower)),
                    BackendBuffer::Cuda(Arc::new(upper)),
                    decomp.row_permutation().to_vec(),
                    decomp.col_permutation().to_vec(),
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
            shape: vec![n, n],
        },
        PyArray {
            buffer: u_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
        row_perm,
        col_perm,
    ))
}
