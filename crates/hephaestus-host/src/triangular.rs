//! Host reference implementor of [`hephaestus_core::TriangularOps`].
//!
//! Walks the two-dimensional index space directly, keeping or zeroing each
//! element via [`hephaestus_core::triangular_keeps`] — the reference
//! substrate favors an obviously correct implementation over a fast one.

use eunomia::Pod;
use hephaestus_core::{
    Result, StridedView, TriangularMode, TriangularOps, triangular_keeps, validate_triangular_shape,
};
use leto_ops::Scalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Host-backed triangular masking for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostTriangularOps;

impl<T> TriangularOps<HostDevice, T> for HostTriangularOps
where
    T: Pod + Scalar,
{
    fn triangular_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        mode: TriangularMode,
        diagonal: i64,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()> {
        validate_triangular_shape(input.layout.shape(), output.layout.shape())?;
        let [rows, cols] = input.layout.shape();

        let in_cells = input.buffer.read();
        let mut out_cells = output.buffer.write();
        for row in 0..rows {
            for col in 0..cols {
                let value = if triangular_keeps(mode, row, col, diagonal) {
                    let offset = input.layout.offset_of([row, col]).map_err(map_leto_error)?;
                    in_cells[offset]
                } else {
                    T::ZERO
                };
                let out_offset = output
                    .layout
                    .offset_of([row, col])
                    .map_err(map_leto_error)?;
                out_cells[out_offset] = value;
            }
        }
        Ok(())
    }
}
