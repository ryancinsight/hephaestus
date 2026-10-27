//! GPU-resident Compressed Sparse Row (CSR) matrices and HIP dispatch.

mod matrix;
mod prepared;
/// Device-neutral sparse operator seam implementation.
pub mod seam;
mod spmm;
mod spmv;

pub use matrix::GpuCsrMatrix;
pub use prepared::{
    PreparedSparseDispatch, PreparedSpmm, PreparedSpmv, prepare_spmm, prepare_spmv,
    prepare_spmv_many, submit_prepared_sparse_batch,
};
pub use spmm::{spmm, spmm_into, spmv_many, spmv_many_into};
pub use spmv::{spmv, spmv_into};
