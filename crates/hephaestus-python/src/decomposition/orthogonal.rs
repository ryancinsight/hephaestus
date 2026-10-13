use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendDevice, clone_cuda_buffer};
use leto::Layout;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

/// Hessenberg reduction on the GPU: returns `(Q, H)` with `A = Q H Qᵀ`,
/// `H` upper-Hessenberg, `Q` orthogonal. Mirrors `scipy.linalg.hessenberg`
/// (which returns `(H, Q)`).
#[pyfunction]
pub(crate) fn hessenberg(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "hessenberg requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (q_buf, h_buf) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::hessenberg(device, op)?;
                Ok((
                    BackendBuffer::Wgpu(decomp.q_buffer().clone()),
                    BackendBuffer::Wgpu(decomp.h_buffer().clone()),
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::hessenberg(device, op)?;
                Ok((
                    clone_cuda_buffer(device, decomp.q_buffer())?,
                    clone_cuda_buffer(device, decomp.h_buffer())?,
                ))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: q_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
        PyArray {
            buffer: h_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
    ))
}

/// Bidiagonalization on the GPU: `A = U B Vᵀ` with `B` upper-bidiagonal,
/// `U`/`V` orthogonal. Returns `(U, B, V)` for `A` of shape `[m, n]`
/// (`U: [m, m]`, `B: [m, n]`, `V: [n, n]`).
#[pyfunction]
pub(crate) fn bidiagonalize(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray, PyArray)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("bidiagonalize requires a 2D matrix"));
    }
    let m = a.shape[0];
    let n = a.shape[1];
    let layout = Layout::c_contiguous([m, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (u_buf, b_buf, v_buf) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::bidiagonalize(device, op)?;
                Ok((
                    BackendBuffer::Wgpu(decomp.u_buffer().clone()),
                    BackendBuffer::Wgpu(decomp.b_buffer().clone()),
                    BackendBuffer::Wgpu(decomp.v_buffer().clone()),
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::bidiagonalize(device, op)?;
                Ok((
                    clone_cuda_buffer(device, decomp.u_buffer())?,
                    clone_cuda_buffer(device, decomp.b_buffer())?,
                    clone_cuda_buffer(device, decomp.v_buffer())?,
                ))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: u_buf,
            device: a.device.clone(),
            shape: vec![m, m],
        },
        PyArray {
            buffer: b_buf,
            device: a.device.clone(),
            shape: vec![m, n],
        },
        PyArray {
            buffer: v_buf,
            device: a.device.clone(),
            shape: vec![n, n],
        },
    ))
}

#[pyfunction]
pub(crate) fn qr(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("qr requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (q_buf, r_buf, q_shape, r_shape) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::qr_decompose_blocked(device, op)?;
                // `inner().q()` is `m`x`m` and `inner().r()` is `m`x`n` by
                // construction, so the reported shapes come from the factored
                // shape rather than from materialising either host factor.
                let (m, n) = decomp.shape();
                let (q_shape, r_shape) = (vec![m, m], vec![m, n]);
                let q_buffer = decomp.accumulate_q(device)?;
                // Neither factor is re-uploaded. R is device-resident and
                // value-identical to `inner().r()` — both entry points leave
                // clean upper-triangular R in the decomposition's buffer — so
                // taking it avoids re-sending `4mn` bytes the device already
                // holds. Q is accumulated on the device from the stored
                // reflectors instead of uploading a host-accumulated `4m^2`.
                Ok((
                    BackendBuffer::Wgpu(q_buffer),
                    BackendBuffer::Wgpu(decomp.into_r_buffer()),
                    q_shape,
                    r_shape,
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::qr_decompose_blocked(device, op)?;
                let (m, n) = decomp.shape();
                let q_buffer = decomp.accumulate_q(device)?;
                // Q is accumulated on the device from the compact
                // Householder representation. R already resides in the
                // cleaned upper triangle of the decomposition buffer, so
                // consuming it avoids both host staging and a device copy.
                Ok((
                    BackendBuffer::Cuda(Arc::new(q_buffer)),
                    BackendBuffer::Cuda(Arc::new(decomp.into_r_buffer())),
                    vec![m, m],
                    vec![m, n],
                ))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: q_buf,
            device: a.device.clone(),
            shape: q_shape,
        },
        PyArray {
            buffer: r_buf,
            device: a.device.clone(),
            shape: r_shape,
        },
    ))
}

#[pyfunction]
pub(crate) fn col_piv_qr(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<u64>)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("col_piv_qr requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (q_buf, r_buf, m, n, perm) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::col_piv_qr(device, op)?;
                let (m, n) = decomp.shape();
                let q_buf = decomp.q().clone();
                let r_buf = decomp.r().clone();
                let perm = decomp.permutation().iter().map(|&x| x as u64).collect();
                Ok((
                    BackendBuffer::Wgpu(q_buf),
                    BackendBuffer::Wgpu(r_buf),
                    m,
                    n,
                    perm,
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::col_piv_qr(device, op)?;
                let (m, n) = decomp.shape();
                let perm = decomp.permutation().iter().map(|&x| x as u64).collect();
                Ok((
                    clone_cuda_buffer(device, decomp.q())?,
                    clone_cuda_buffer(device, decomp.r())?,
                    m,
                    n,
                    perm,
                ))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
