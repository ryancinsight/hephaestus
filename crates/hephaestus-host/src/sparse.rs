//! Leto as a sparse-operator-seam implementor (ADR 0046, ADR 0048).
//!
//! [`HostSparseOps`] holds a matrix as leto-ops' [`CsrMatrix`](leto_ops::CsrMatrix), whose
//! constructor validates every CSR invariant the seam documents, and applies
//! it through leto-ops' `spmv_into`/`spmm_into`. The conformance suite's
//! sparse-operator and batch-submit clauses run on the host pair.
//!
//! Shapes and aliasing are validated before any output element is written.

use eunomia::Pod;
use hephaestus_core::{BatchSubmitOps, HephaestusError, Result, SparseOperatorOps, StridedView};
use leto::{ArrayView, Layout};
use leto_ops::{CsrMatrix, Scalar};

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Sparse matrix application for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostSparseOps;

/// An SpMV bound to its matrix, input, and output.
///
/// The host dispatches no kernel, so preparation validates the operands once
/// and binds their identities; each dispatch re-reads the bound input.
pub struct HostPreparedApply<'op, T> {
    matrix: &'op CsrMatrix<T>,
    input: &'op HostBuffer<T>,
    output: &'op HostBuffer<T>,
}

/// Reject an operand pair naming one allocation: the output would be written
/// under the lock the input is read through.
fn require_distinct<T>(input: &HostBuffer<T>, output: &HostBuffer<T>) -> Result<()> {
    if input.aliases(output) {
        return Err(HephaestusError::DispatchFailed {
            message: "sparse output buffer must not alias its input".to_string(),
        });
    }
    Ok(())
}

/// Reject a vector whose length differs from the matrix dimension it meets.
fn require_len(expected: usize, actual: usize) -> Result<()> {
    HostDevice::require_matching_len(expected, actual)
}

/// Validate an SpMV's operands against the matrix shape.
fn validate_apply<T: Pod + Scalar>(
    matrix: &CsrMatrix<T>,
    input: &HostBuffer<T>,
    output: &HostBuffer<T>,
) -> Result<()> {
    require_distinct(input, output)?;
    let (rows, columns) = matrix.shape();
    require_len(columns, input.read().len())?;
    require_len(rows, output.read().len())
}

/// `output = matrix · input` over validated operands.
fn spmv<T: Pod + Scalar>(
    matrix: &CsrMatrix<T>,
    input: &HostBuffer<T>,
    output: &HostBuffer<T>,
) -> Result<()> {
    let input_cells = input.read();
    let layout = Layout::c_contiguous([input_cells.len()]).map_err(map_leto_error)?;
    let input_view = ArrayView::try_new(layout, &input_cells).map_err(map_leto_error)?;
    leto_ops::spmv_into(matrix, &input_view, &mut output.write()).map_err(map_leto_error)
}

impl<T> SparseOperatorOps<HostDevice, T> for HostSparseOps
where
    T: Pod + Scalar,
{
    type Matrix = CsrMatrix<T>;
    type PreparedApply<'op>
        = HostPreparedApply<'op, T>
    where
        Self: 'op,
        HostDevice: 'op,
        T: 'op;

    fn upload_csr(
        &self,
        _device: &HostDevice,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self::Matrix> {
        CsrMatrix::from_parts(
            values.to_vec(),
            col_indices.to_vec(),
            row_ptr.to_vec(),
            rows,
            columns,
        )
        .map_err(map_leto_error)
    }

    fn shape(&self, matrix: &Self::Matrix) -> (usize, usize) {
        matrix.shape()
    }

    fn apply(
        &self,
        _device: &HostDevice,
        matrix: &Self::Matrix,
        input: &HostBuffer<T>,
        output: &mut HostBuffer<T>,
    ) -> Result<()> {
        validate_apply(matrix, input, output)?;
        spmv(matrix, input, output)
    }

    fn apply_batch(
        &self,
        _device: &HostDevice,
        matrix: &Self::Matrix,
        batch: StridedView<'_, HostBuffer<T>, 2>,
        output: &mut HostBuffer<T>,
    ) -> Result<()> {
        require_distinct(batch.buffer, output)?;
        let (rows, columns) = matrix.shape();
        let [batch_rows, batch_columns] = batch.layout.shape();
        require_len(columns, batch_rows)?;
        let product_len = rows.checked_mul(batch_columns).ok_or_else(|| {
            HephaestusError::InvalidConfiguration {
                message: format!(
                    "SpMM output of {rows} rows by {batch_columns} columns overflows usize"
                ),
            }
        })?;
        require_len(product_len, output.read().len())?;
        let batch_cells = batch.buffer.read();
        let batch_view = ArrayView::try_new(*batch.layout, &batch_cells).map_err(map_leto_error)?;
        leto_ops::spmm_into(matrix, &batch_view, &mut output.write()).map_err(map_leto_error)
    }

    fn nnz(&self, matrix: &Self::Matrix) -> usize {
        matrix.nnz()
    }

    fn prepare_apply<'op>(
        &self,
        _device: &'op HostDevice,
        matrix: &'op Self::Matrix,
        input: &'op HostBuffer<T>,
        output: &'op HostBuffer<T>,
    ) -> Result<Self::PreparedApply<'op>> {
        validate_apply(matrix, input, output)?;
        Ok(HostPreparedApply {
            matrix,
            input,
            output,
        })
    }

    fn dispatch_apply(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedApply<'_>,
    ) -> Result<()> {
        spmv(prepared.matrix, prepared.input, prepared.output)
    }
}

impl<T> BatchSubmitOps<HostDevice, T> for HostSparseOps
where
    T: Pod + Scalar,
{
    type Dispatch<'op>
        = &'op HostPreparedApply<'op, T>
    where
        Self: 'op,
        HostDevice: 'op,
        T: 'op;

    fn spmv_dispatch<'plan, 'op: 'plan>(
        &self,
        prepared: &'plan Self::PreparedApply<'op>,
    ) -> Self::Dispatch<'plan> {
        prepared
    }

    /// The host has no submission to amortize, so the batch runs each
    /// dispatch in order: the seam's result-equivalence contract, with no
    /// timing claim.
    fn submit_batch(&self, device: &HostDevice, operations: &[Self::Dispatch<'_>]) -> Result<()> {
        operations
            .iter()
            .try_for_each(|prepared| self.dispatch_apply(device, prepared))
    }
}
