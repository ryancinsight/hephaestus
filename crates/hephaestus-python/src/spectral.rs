//! Spectral routines: SVD, symmetric eigendecomposition, singular
//! values, Schur form, and general eigenvalues.

use crate::array::PyArray;
use crate::backend::{BackendBuffer, BackendComplexBuffer, BackendDevice, clone_cuda_buffer};
use eunomia::{Complex, Complex32};
use hephaestus_core::ComputeDevice;
use hephaestus_cuda::CudaDevice;
use hephaestus_wgpu::WgpuDevice;
use leto::Layout;
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

/// Host-marshaled SVD factors and their shapes (NumPy/SciPy `Vt` convention).
struct SvdHostOutputs {
    u: Vec<f32>,
    s: Vec<f32>,
    vt: Vec<f32>,
    u_shape: Vec<usize>,
    s_shape: Vec<usize>,
    vt_shape: Vec<usize>,
}

/// Device-resident SVD factors and their shapes.
struct SvdDeviceOutputs {
    u: BackendBuffer,
    s: BackendBuffer,
    vt: BackendBuffer,
    u_shape: Vec<usize>,
    s_shape: Vec<usize>,
    vt_shape: Vec<usize>,
}

/// Marshal a host SVD result into flat vectors and shapes for upload.
///
/// Pure layout marshaling, no factor arithmetic: `U` and the singular
/// values are copied out as-is, and `V` is transposed to the NumPy/SciPy
/// `Vt` output convention (an index permutation, not a mathematical
/// transformation of the decomposition). It therefore stays in the
/// binding layer rather than `hephaestus-core`.
fn svd_host_outputs(
    inner: &leto_ops::SvdDecomposition<f32>,
) -> hephaestus_core::Result<SvdHostOutputs> {
    let u_host = inner.left_singular_vectors.clone();
    let s_host =
        leto::Array1::from_shape_vec([inner.singular_values.len()], inner.singular_values.clone())
            .map_err(|e| hephaestus_core::HephaestusError::DispatchFailed {
                message: e.to_string(),
            })?;
    let vt_transposed = inner
        .right_singular_vectors
        .transpose([1, 0])
        .map_err(|e| hephaestus_core::HephaestusError::DispatchFailed {
            message: e.to_string(),
        })?;
    let vt_host = vt_transposed.to_contiguous();
    Ok(SvdHostOutputs {
        u: leto::Storage::as_slice(u_host.storage()).to_vec(),
        s: leto::Storage::as_slice(s_host.storage()).to_vec(),
        vt: leto::Storage::as_slice(vt_host.storage()).to_vec(),
        u_shape: vec![u_host.shape()[0], u_host.shape()[1]],
        s_shape: vec![s_host.shape()[0]],
        vt_shape: vec![vt_host.shape()[0], vt_host.shape()[1]],
    })
}

fn upload_svd_outputs_wgpu(
    device: &WgpuDevice,
    inner: &leto_ops::SvdDecomposition<f32>,
) -> hephaestus_core::Result<SvdDeviceOutputs> {
    let host = svd_host_outputs(inner)?;
    Ok(SvdDeviceOutputs {
        u: BackendBuffer::Wgpu(device.upload(&host.u)?),
        s: BackendBuffer::Wgpu(device.upload(&host.s)?),
        vt: BackendBuffer::Wgpu(device.upload(&host.vt)?),
        u_shape: host.u_shape,
        s_shape: host.s_shape,
        vt_shape: host.vt_shape,
    })
}

fn upload_svd_outputs_cuda(
    device: &CudaDevice,
    inner: &leto_ops::SvdDecomposition<f32>,
) -> hephaestus_core::Result<SvdDeviceOutputs> {
    let host = svd_host_outputs(inner)?;
    Ok(SvdDeviceOutputs {
        u: BackendBuffer::Cuda(Arc::new(device.upload(&host.u)?)),
        s: BackendBuffer::Cuda(Arc::new(device.upload(&host.s)?)),
        vt: BackendBuffer::Cuda(Arc::new(device.upload(&host.vt)?)),
        u_shape: host.u_shape,
        s_shape: host.s_shape,
        vt_shape: host.vt_shape,
    })
}

#[pyfunction]
pub(crate) fn svd(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray, PyArray)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err("svd requires a 2D matrix"));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let outputs = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::svd_decompose(device, op)?;
                upload_svd_outputs_wgpu(device, decomp.inner())
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::svd_decompose(device, op)?;
                upload_svd_outputs_cuda(device, decomp.inner())
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: outputs.u,
            device: a.device.clone(),
            shape: outputs.u_shape,
        },
        PyArray {
            buffer: outputs.s,
            device: a.device.clone(),
            shape: outputs.s_shape,
        },
        PyArray {
            buffer: outputs.vt,
            device: a.device.clone(),
            shape: outputs.vt_shape,
        },
    ))
}

#[pyfunction]
pub(crate) fn symmetric_eigen(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "symmetric_eigen requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (w_buf, v_buf, w_shape, v_shape) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::symmetric_eigen_jacobi(device, op)?;
                let w_host = &decomp.inner().eigenvalues;
                let v_host = decomp.inner().eigenvectors.clone();
                Ok((
                    BackendBuffer::Wgpu(device.upload(w_host)?),
                    BackendBuffer::Wgpu(device.upload(leto::Storage::as_slice(v_host.storage()))?),
                    vec![w_host.len()],
                    vec![v_host.shape()[0], v_host.shape()[1]],
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::symmetric_eigen_jacobi(device, op)?;
                let w_host = &decomp.inner().eigenvalues;
                let v_host = decomp.inner().eigenvectors.clone();
                Ok((
                    BackendBuffer::Cuda(Arc::new(device.upload(w_host)?)),
                    BackendBuffer::Cuda(Arc::new(
                        device.upload(leto::Storage::as_slice(v_host.storage()))?,
                    )),
                    vec![w_host.len()],
                    vec![v_host.shape()[0], v_host.shape()[1]],
                ))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray {
            buffer: w_buf,
            device: a.device.clone(),
            shape: w_shape,
        },
        PyArray {
            buffer: v_buf,
            device: a.device.clone(),
            shape: v_shape,
        },
    ))
}

#[pyfunction]
pub(crate) fn singular_values(py: Python<'_>, a: &PyArray) -> PyResult<PyArray> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err(
            "singular_values requires a 2D matrix",
        ));
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let layout =
        Layout::c_contiguous([rows, cols]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let s_buf = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                hephaestus_wgpu::singular_values(
                    device,
                    hephaestus_wgpu::StridedOperand {
                        buffer,
                        layout: &layout,
                    },
                )
                .map(BackendBuffer::Wgpu)
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                hephaestus_cuda::singular_values(
                    device,
                    hephaestus_cuda::StridedOperand {
                        buffer,
                        layout: &layout,
                    },
                )
                .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray {
        buffer: s_buf,
        device: a.device.clone(),
        shape: vec![rows.min(cols)],
    })
}

#[pyfunction]
pub(crate) fn schur(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err("schur requires a square 2D matrix"));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let (q_buf, t_buf, n_val) = py
        .detach(move || match (&dev, &buf) {
            (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                let op = hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_wgpu::schur(device, op)?;
                Ok((
                    BackendBuffer::Wgpu(decomp.q_buffer().clone()),
                    BackendBuffer::Wgpu(decomp.t_buffer().clone()),
                    decomp.n(),
                ))
            }
            (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                let op = hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &layout,
                };
                let decomp = hephaestus_cuda::schur(device, op)?;
                Ok((
                    clone_cuda_buffer(device, decomp.q_buffer())?,
                    clone_cuda_buffer(device, decomp.t_buffer())?,
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
pub(crate) fn eigenvalues<'py>(
    py: Python<'py>,
    a: &PyArray,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(PyValueError::new_err(
            "eigenvalues requires a square 2D matrix",
        ));
    }
    let n = a.shape[0];
    let layout = Layout::c_contiguous([n, n]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dev = a.device.clone();
    let buf = a.buffer.clone();

    let host_data = py
        .detach(move || {
            let mut host_data = vec![Complex::new(0.0f32, 0.0f32); n];
            let e_buf = match (&dev, &buf) {
                (BackendDevice::Wgpu(device), BackendBuffer::Wgpu(buffer)) => {
                    hephaestus_wgpu::eigenvalues(
                        device,
                        hephaestus_wgpu::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(BackendComplexBuffer::Wgpu)?
                }
                (BackendDevice::Cuda(device), BackendBuffer::Cuda(buffer)) => {
                    hephaestus_cuda::eigenvalues(
                        device,
                        hephaestus_cuda::StridedOperand {
                            buffer,
                            layout: &layout,
                        },
                    )
                    .map(BackendComplexBuffer::Cuda)?
                }
                _ => {
                    return Err(hephaestus_core::HephaestusError::DispatchFailed {
                        message: "array buffer belongs to a different backend".to_string(),
                    });
                }
            };
            dev.download_complex(&e_buf, &mut host_data)?;
            Ok::<_, hephaestus_core::HephaestusError>(host_data)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray1::from_vec(py, host_data))
}
