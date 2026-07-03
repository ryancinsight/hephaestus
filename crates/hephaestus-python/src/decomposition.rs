//! Matrix factorisations: Cholesky, LU (partial and full pivoting), QR
//! (plain and column-pivoted), Hessenberg, bidiagonalisation, and
//! Bunch-Kaufman. Factor-splitting math lives in `hephaestus-core`
//! (`split_packed_lu`); this module only marshals buffers.

use crate::array::PyArray;
use crate::backend::{clone_cuda_buffer, BackendBuffer, BackendDevice};
use hephaestus_core::{split_packed_lu, ComputeDevice, DeviceBuffer};
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
        .allow_threads(move || match (&dev, &buf) {
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
pub(crate) fn lu(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray, Vec<usize>)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("lu requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (decomp, l_buf, u_buf) = py
        .allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::lu_decompose_blocked(device, op)?;
                let mut host_factors = vec![0.0f32; n * n];
                device.download(decomp.factors(), &mut host_factors)?;
                let (host_l, host_u) = split_packed_lu(&host_factors, n)?;
                let l_buf = BackendBuffer::Wgpu(device.upload(&host_l)?);
                let u_buf = BackendBuffer::Wgpu(device.upload(&host_u)?);
                Ok((decomp.pivots().to_vec(), l_buf, u_buf))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::lu_decompose_blocked(device, op)?;
                let mut host_factors = vec![0.0f32; n * n];
                device.download(decomp.factors(), &mut host_factors)?;
                let (host_l, host_u) = split_packed_lu(&host_factors, n)?;
                let l_buf = BackendBuffer::Cuda(Arc::new(device.upload(&host_l)?));
                let u_buf = BackendBuffer::Cuda(Arc::new(device.upload(&host_u)?));
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
        .allow_threads(move || match (&dev, &buf) {
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
        .allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::full_piv_lu(device, op)?;
                let mut host_factors = vec![0.0f32; n * n];
                device.download(decomp.lu_buffer(), &mut host_factors)?;
                let (host_l, host_u) = split_packed_lu(&host_factors, n)?;
                Ok((
                    BackendBuffer::Wgpu(device.upload(&host_l)?),
                    BackendBuffer::Wgpu(device.upload(&host_u)?),
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
                let mut host_factors = vec![0.0f32; n * n];
                device.download(decomp.lu_buffer(), &mut host_factors)?;
                let (host_l, host_u) = split_packed_lu(&host_factors, n)?;
                Ok((
                    BackendBuffer::Cuda(Arc::new(device.upload(&host_l)?)),
                    BackendBuffer::Cuda(Arc::new(device.upload(&host_u)?)),
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
        .allow_threads(move || match (&dev, &buf) {
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
        .allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::qr_decompose_blocked(device, op)?;
                let q_host = decomp.inner().q();
                let r_host = decomp.inner().r();
                Ok((
                    BackendBuffer::Wgpu(device.upload(leto::Storage::as_slice(q_host.storage()))?),
                    BackendBuffer::Wgpu(device.upload(leto::Storage::as_slice(r_host.storage()))?),
                    vec![q_host.shape()[0], q_host.shape()[1]],
                    vec![r_host.shape()[0], r_host.shape()[1]],
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::qr_decompose_blocked(device, op)?;
                let q_host = decomp.inner().q();
                let r_host = decomp.inner().r();
                Ok((
                    BackendBuffer::Cuda(Arc::new(
                        device.upload(leto::Storage::as_slice(q_host.storage()))?,
                    )),
                    BackendBuffer::Cuda(Arc::new(
                        device.upload(leto::Storage::as_slice(r_host.storage()))?,
                    )),
                    vec![q_host.shape()[0], q_host.shape()[1]],
                    vec![r_host.shape()[0], r_host.shape()[1]],
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
        .allow_threads(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::col_piv_qr(device, op)?;
                let q_buf = decomp.q().clone();
                let r_buf = decomp.r().clone();
                let m = (q_buf.len() as f64).sqrt() as usize;
                let n = r_buf.len().checked_div(m).unwrap_or(0);
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
                let q_len = decomp.q().len();
                let r_len = decomp.r().len();
                let m = (q_len as f64).sqrt() as usize;
                let n = r_len.checked_div(m).unwrap_or(0);
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
        .allow_threads(move || match (&dev, &buf) {
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
