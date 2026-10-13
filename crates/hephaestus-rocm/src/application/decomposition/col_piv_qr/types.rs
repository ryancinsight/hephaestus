use eunomia::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};
use leto::Layout;

use crate::RocmDevice;
use crate::infrastructure::RocmBuffer;

pub(crate) const WIDTH: BlockWidth = BlockWidth::DEFAULT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ColPivQrMeta {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) pivots: u32,
    pub(crate) k: u32,
}

const _: () = assert!(core::mem::size_of::<ColPivQrMeta>() == 16);

/// Column-pivoted QR result with device-backed **Q**, **R**, and permutation.
pub struct GpuColPivQrDecomposition {
    /// Host-side reference retained for scalar least-squares solves.
    pub(crate) inner: leto_ops::ColPivQrDecomposition<f32>,
    /// Materialized orthogonal factor **Q** on the device.
    pub(crate) q: RocmBuffer<f32>,
    /// Upper-triangular factor **R** on the device.
    pub(crate) r: RocmBuffer<f32>,
    /// Column permutation, where each entry is the original column at that position.
    pub(crate) permutation: Vec<usize>,
    pub(crate) rank: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl GpuColPivQrDecomposition {
    /// Numerical rank reported by the pivoted diagonal threshold.
    #[must_use]
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `(rows, cols)` of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the orthogonal factor **Q** buffer on the device.
    #[must_use]
    #[inline]
    pub fn q(&self) -> &RocmBuffer<f32> {
        &self.q
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r(&self) -> &RocmBuffer<f32> {
        &self.r
    }

    /// Return the column permutation.
    #[must_use]
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ using the retained factor.
    pub fn solve_least_squares(
        &self,
        device: &RocmDevice,
        rhs: &RocmBuffer<f32>,
    ) -> Result<RocmBuffer<f32>> {
        if rhs.len() != self.rows {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.rows,
                device_len: rhs.len(),
            });
        }
        if self.rows == 0 || self.cols == 0 {
            return device.upload(&[] as &[f32]);
        }

        let rhs_host = device.download_owned(rhs)?;
        let rhs_view = leto::ArrayView::<f32, 1>::new(
            Layout::c_contiguous([self.rows]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution = self.inner.solve_least_squares(&rhs_view).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR least-squares solve failed: {error}"),
            }
        })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }
}
