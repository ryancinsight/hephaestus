//! GPU-resident Compressed Sparse Row (CSR) matrix representation.

mod batch;
mod matrix;
mod seam;
mod spmm;
mod spmv;

pub use batch::{PreparedSparseDispatch, submit_prepared_sparse_batch};
pub use matrix::GpuCsrMatrix;
pub use seam::WgpuSparseOps;
pub use spmm::{
    PreparedSpmm, prepare_spmm, prepare_spmv_many, spmm, spmm_into, spmv_many, spmv_many_into,
};
pub use spmv::{PreparedSpmv, prepare_spmv, spmv, spmv_into};
