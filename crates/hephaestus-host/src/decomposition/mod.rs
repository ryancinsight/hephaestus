//! Leto as a decomposition-seam implementor (ADR 0046 / ADR 0039 §3).
//!
//! [`HostDecompositionOps`] adapts leto-ops' fifteen decomposition entry
//! points onto [`DecompositionOps<HostDevice>`](hephaestus_core::DecompositionOps), so the CPU substrate joins
//! the same role trait the GPU backends implement and the conformance
//! suite's decomposition clauses run on the host pair. Factor storage
//! conventions follow the seam contracts pinned by the clause suite
//! (packed LU factors, m×n row-major `R`, column-convention singular and
//! eigen vectors, dense block-diagonal Bunch–Kaufman `D`, length-`n` UDU
//! diagonal).

mod handles;
mod ops;

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Layout;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Dense decompositions for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDecompositionOps;

pub(super) fn buffer_of<T: Pod>(device: &HostDevice, values: &[T]) -> Result<HostBuffer<T>> {
    device.upload(values)
}

/// Solve through a leto handle: download the rhs, apply, re-upload.
pub(super) fn solve_with<E: core::fmt::Display>(
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
    buffer_of(device, leto::Storage::as_slice(solution.storage()))
}

pub use handles::{
    HostBidiagonal, HostBunchKaufman, HostCholesky, HostColPivQr, HostFullPivLu, HostHessenberg,
    HostLu, HostQr, HostSchur, HostSvd, HostSymmetricEigen, HostUdu,
};
