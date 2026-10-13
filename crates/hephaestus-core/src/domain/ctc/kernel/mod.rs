//! Shared CTC step kernels for flat-argument backends.
//!
//! One monomorphized implementation serves CUDA C and HIP C: the alpha and
//! beta frame steps, the likelihood finish, and the posterior sweep,
//! dispatched through the [`MultiStorageKernel`](crate::MultiStorageKernel)
//! seam with the shared [`source`] text. Backends instantiate
//! `CtcKernel` with their own kernel type and bind the step traits to it.

mod dispatch;
mod source;

pub use dispatch::CtcKernel;
