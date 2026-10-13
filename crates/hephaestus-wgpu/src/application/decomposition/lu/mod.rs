//! GPU-resident LU decomposition with partial pivoting.
//!
//! Computes **P A** = **L U** where **P** is a permutation matrix, **L** is
//! unit lower-triangular, and **U** is upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`lu_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`lu_decompose_blocked`] — blocked algorithm where panel
//!   factorization uses the same partial-pivoting rule as Leto's LU and the
//!   O(2n³/3) trailing GEMM update (`A₂₂ -= L₂₁ · U₁₂`) runs on the GPU via a
//!   dedicated compute kernel.
//!
//! # Mathematical Foundations
//!
//! ## Blocked LU with GPU Trailing GEMM
//!
//! Partition **P A = L U** into *b × b* blocks:
//!
//! ```text
//! ┌           ┐   ┌       ┐ ┌       ┐
//! │ A₁₁  A₁₂ │   │ L₁₁ 0 │ │ U₁₁ U₁₂│
//! │ A₂₁  A₂₂ │ = │ L₂₁ I │ │  0  S₂₂│
//! └           ┘   └       ┘ └       ┘
//! ```
//!
//! The Schur complement is **S₂₂ = A₂₂ − L₂₁ U₁₂** and the dominant
//! cost is the rank-b GEMM update which runs on the GPU.
//!
//! **Theorem (Blocked LU complexity).** For *n × n* with block size *b*,
//! the total flop count is 2n³/3, identical to unblocked LU.  The blocked
//! variant improves performance by:
//! (a) moving the O(b(n−k)²) trailing GEMM to the GPU, and
//! (b) improving CPU cache locality for the O(b²(n−k)) panel operations.
//!
//! **Proof.** Each block iteration costs:
//! - Panel factor: 2b³/3
//! - Panel solve (L₂₁): b²(n−k−b)/2
//! - Panel solve (U₁₂): b²(n−k−b)/2
//! - Trailing GEMM: 2b(n−k−b)²
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n³/3 total flops (the same
//! as unblocked LU).  The key performance gain is that the trailing GEMM,
//! which dominates for large n, executes on the GPU's massively parallel
//! compute units rather than on the CPU's sequential cores.

mod api;
mod gemm;
mod types;

pub use api::{lu_decompose, lu_decompose_blocked};
pub(crate) use gemm::{GemmTrailingUpdate, gemm_trailing_update};
pub use types::GpuLuDecomposition;
