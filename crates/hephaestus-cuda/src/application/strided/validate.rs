//! Strided layout validation and broadcast.

use super::types::StridedLayout;
use super::{MAX_STRIDED_RANK, map_layout_err, to_u32};
use crate::infrastructure::buffer::CudaBuffer;
use hephaestus_core::{DeviceBuffer, HephaestusError, Result};
use leto::Layout;

pub(crate) fn validate_out<T, const N: usize>(
    out: &CudaBuffer<T>,
    out_layout: &Layout<N>,
) -> Result<usize> {
    if out_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    out_layout
        .validate_storage_len(out.len())
        .map_err(map_layout_err)?;
    out_layout.checked_size().map_err(map_layout_err)
}

pub(crate) fn validate_dyn_layout_shape(layout: StridedLayout<'_>) -> Result<()> {
    if layout.shape.len() != layout.strides.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "layout rank mismatch: shape rank {}, stride rank {}",
                layout.shape.len(),
                layout.strides.len()
            ),
        });
    }
    if layout.shape.len() > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "strided dispatch supports rank <= {MAX_STRIDED_RANK}, got {}",
                layout.shape.len()
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_dyn_layout<T>(
    what: &str,
    buffer: &CudaBuffer<T>,
    layout: StridedLayout<'_>,
) -> Result<usize> {
    validate_dyn_layout_shape(layout)?;
    let len = layout
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("{what} logical size overflows usize"),
        })?;
    if len == 0 {
        return Ok(0);
    }

    let max_offset = layout
        .shape
        .iter()
        .zip(layout.strides)
        .try_fold(layout.offset, |acc, (&dim, &stride)| {
            dim.checked_sub(1)
                .and_then(|span| span.checked_mul(stride))
                .and_then(|span| acc.checked_add(span))
        })
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("{what} layout address range overflows usize"),
        })?;
    if max_offset >= buffer.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: max_offset + 1,
            device_len: buffer.len(),
        });
    }
    Ok(len)
}

pub(crate) fn validate_dyn_out<T>(out: &CudaBuffer<T>, layout: StridedLayout<'_>) -> Result<usize> {
    if layout
        .shape
        .iter()
        .zip(layout.strides)
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
    {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    validate_dyn_layout("output", out, layout)
}

#[derive(Clone, Copy)]
pub(crate) struct BroadcastDyn {
    pub(crate) strides: [i32; 8],
    pub(crate) offset: u32,
}

pub(crate) fn broadcast_dyn_layout(
    layout: StridedLayout<'_>,
    out_shape: &[usize],
) -> Result<BroadcastDyn> {
    validate_dyn_layout_shape(layout)?;
    if out_shape.len() > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "strided dispatch supports rank <= {MAX_STRIDED_RANK}, got {}",
                out_shape.len()
            ),
        });
    }
    if layout.shape.len() > out_shape.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "cannot broadcast rank {} layout to rank {} output",
                layout.shape.len(),
                out_shape.len()
            ),
        });
    }

    let mut strides = [0i32; 8];
    let rank_delta = out_shape.len() - layout.shape.len();
    for (out_axis, &out_dim) in out_shape.iter().enumerate() {
        let dst = 8 - out_shape.len() + out_axis;
        if out_axis < rank_delta {
            strides[dst] = 0;
            continue;
        }

        let in_axis = out_axis - rank_delta;
        let in_dim = layout.shape[in_axis];
        if in_dim == out_dim {
            strides[dst] = i32::try_from(layout.strides[in_axis]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("stride {} exceeds i32 range", layout.strides[in_axis]),
                }
            })?;
        } else if in_dim == 1 {
            strides[dst] = 0;
        } else {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "incompatible broadcast: input shape {:?} to output shape {:?}",
                    layout.shape, out_shape
                ),
            });
        }
    }

    Ok(BroadcastDyn {
        strides,
        offset: to_u32(layout.offset, "input offset")?,
    })
}
