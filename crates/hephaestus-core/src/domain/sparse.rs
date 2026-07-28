//! Device-neutral sparse operator contracts.

use bytemuck::Pod;

use super::device::ComputeDevice;
use super::error::Result;

/// Backend-neutral sparse matrix-vector application.
///
/// # Why this seam exists
///
/// Every backend crate carries its own `GpuCsrMatrix` and `spmv_into`, taking
/// that backend's concrete device type — the same shape the dense vector
/// operations had before [`super::vector::DenseVectorOps`]. A consumer holding
/// a device-resident operator therefore had to bind to one device API, which is
/// what kept accelerator solver code per-device. This trait closes that gap for
/// the sparse half.
///
/// # Shape
///
/// `Self` is a backend-provided bundle of prepared sparse kernels, constructed
/// once, with the device passed per call — matching
/// [`super::vector::DenseVectorOps`] so a consumer holds one object per
/// concern rather than one per operation.
///
/// # CSR contract
///
/// The upload takes canonical CSR parts rather than any host matrix type, so
/// this crate stays free of a sparse-storage dependency and any producer of
/// CSR can feed it. `row_ptr` has `rows + 1` entries, is non-decreasing, starts
/// at zero and ends at `values.len()`; column indices within a row are strictly
/// increasing and below `columns`. Implementations validate these and return
/// [`crate::HephaestusError`] rather than dispatching against malformed
/// structure.
pub trait SparseOperatorOps<D: ComputeDevice, T: Pod> {
    /// Device-resident sparse matrix.
    type Matrix;

    /// Upload canonical CSR parts to the device.
    ///
    /// # Errors
    ///
    /// Returns a structural CSR violation, an index that does not fit the
    /// backend's native index width, or an allocation or transfer failure.
    fn upload_csr(
        &self,
        device: &D,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self::Matrix>;

    /// Row and column counts of a device-resident matrix.
    fn shape(&self, matrix: &Self::Matrix) -> (usize, usize);

    /// Apply `output = matrix · input`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch against the matrix shape, or the backend
    /// dispatch failure.
    fn apply(
        &self,
        device: &D,
        matrix: &Self::Matrix,
        input: &D::Buffer<T>,
        output: &mut D::Buffer<T>,
    ) -> Result<()>;
}
