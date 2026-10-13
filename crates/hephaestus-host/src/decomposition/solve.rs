use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Layout;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Solve through a leto handle: download the rhs, apply, re-upload.
pub(crate) fn solve_with<E: core::fmt::Display>(
    device: &HostDevice,
    rhs: &HostBuffer<f32>,
    solve: impl FnOnce(leto::ArrayView<'_, f32, 1>) -> core::result::Result<leto::Array1<f32>, E>,
) -> Result<HostBuffer<f32>> {
    let cells = rhs.read();
    let layout =
        Layout::c_contiguous([cells.len()]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("host rhs layout rejected: {error}"),
        })?;
    let view = leto::ArrayView::<f32, 1>::try_new(layout, &cells).map_err(map_leto_error)?;
    let solution = solve(view).map_err(map_leto_error)?;
    device.upload(leto::Storage::as_slice(solution.storage()))
}
