//! Device-neutral sparse operator contracts.

use bytemuck::Pod;

use super::device::ComputeDevice;
use super::error::Result;
use super::view::StridedView;

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
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`](crate::KernelDialect::IEEE_SPECIAL_VALUES)
/// (ADR 0043) for what is and is not promised per dialect.
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

    /// Apply the operator to a dense right-hand-side batch:
    /// `output = matrix · batch`, where `batch` is a rank-2 view whose row
    /// count matches the matrix columns and whose columns are independent
    /// right-hand sides. `output` is row-major `rows × batch-columns`. This
    /// is the SpMM kernel; batched SpMV is the same dispatch.
    ///
    /// # Errors
    ///
    /// Returns a shape or length mismatch against the matrix shape, an
    /// aliased output, or the backend dispatch failure.
    fn apply_batch(
        &self,
        device: &D,
        matrix: &Self::Matrix,
        batch: StridedView<'_, D::Buffer<T>, 2>,
        output: &mut D::Buffer<T>,
    ) -> Result<()>;

    /// Number of explicitly stored values.
    fn nnz(&self, matrix: &Self::Matrix) -> usize;

    /// Prepared SpMV bound to fixed matrix/input/output allocations.
    ///
    /// Lifetime-parameterized so backends whose prepared form borrows the
    /// operands implement the seam without erasing that borrow;
    /// handle-based backends simply ignore `'op`.
    type PreparedApply<'op>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Bind dispatch resources for `output = matrix · input`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch against the matrix shape, an aliasing
    /// violation, or the backend preparation failure.
    fn prepare_apply<'op>(
        &self,
        device: &'op D,
        matrix: &'op Self::Matrix,
        input: &'op D::Buffer<T>,
        output: &'op D::Buffer<T>,
    ) -> Result<Self::PreparedApply<'op>>;

    /// Re-dispatch a prepared SpMV over its bound operands. Writes to the
    /// bound input between dispatches are observed (the rebind contract).
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn dispatch_apply(&self, device: &D, prepared: &Self::PreparedApply<'_>) -> Result<()>;
}

/// Batched submission of prepared sparse dispatches (ADR 0045).
///
/// One submission amortizes per-dispatch overhead: WGPU encodes the batch
/// into a single command buffer, CUDA/HIP launch back-to-back without
/// intervening synchronization. The contract is result equivalence with
/// dispatching each operation individually, in order — not timing — so a
/// backend with no batching advantage may loop.
pub trait BatchSubmitOps<D: ComputeDevice, T: Pod>: SparseOperatorOps<D, T> {
    /// One batchable prepared dispatch: the backend's union of prepared
    /// forms encodable into a single submission.
    type Dispatch<'op>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Wrap a prepared SpMV as a batchable dispatch.
    ///
    /// The dispatch borrows the prepared form (`'plan`), which may be
    /// shorter than the operand borrows inside it (`'op`).
    fn spmv_dispatch<'plan, 'op: 'plan>(
        &self,
        prepared: &'plan Self::PreparedApply<'op>,
    ) -> Self::Dispatch<'plan>;

    /// Submit every operation in order with one device round-trip. An
    /// empty batch is a valid no-op.
    ///
    /// # Errors
    ///
    /// Returns a cross-device batch rejection or the backend submission
    /// failure.
    fn submit_batch(&self, device: &D, operations: &[Self::Dispatch<'_>]) -> Result<()>;
}

/// Validate canonical CSR structure before anything is uploaded.
///
/// The seam accepts raw parts, so the invariants a host matrix type would
/// have enforced at construction are checked here instead: dispatching
/// against malformed structure would read out of bounds on the device.
///
/// # Errors
///
/// Returns a typed dispatch error naming the violated CSR invariant.
pub fn validate_csr<T>(
    values: &[T],
    col_indices: &[usize],
    row_ptr: &[usize],
    rows: usize,
    columns: usize,
) -> crate::Result<()> {
    let malformed = |reason: String| crate::HephaestusError::DispatchFailed {
        message: format!("invalid CSR: {reason}"),
    };
    if row_ptr.len() != rows + 1 {
        return Err(malformed(format!(
            "row_ptr length {} must be rows + 1 = {}",
            row_ptr.len(),
            rows + 1
        )));
    }
    if col_indices.len() != values.len() {
        return Err(malformed(format!(
            "col_indices length {} differs from values length {}",
            col_indices.len(),
            values.len()
        )));
    }
    if row_ptr[0] != 0 || row_ptr[rows] != values.len() {
        return Err(malformed(
            "row_ptr must start at 0 and end at the nonzero count".to_owned(),
        ));
    }
    for window in row_ptr.windows(2) {
        if window[0] > window[1] {
            return Err(malformed("row_ptr must be non-decreasing".to_owned()));
        }
        let row = &col_indices[window[0]..window[1]];
        if row.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(malformed(
                "column indices in each row must be strictly increasing".to_owned(),
            ));
        }
        if row.last().is_some_and(|&last| last >= columns) {
            return Err(malformed(format!(
                "column index out of range for {columns} columns"
            )));
        }
    }
    Ok(())
}
