//! One integration-test binary for the host crate: each module instantiates
//! a shared conformance clause, or pins a host-specific case, for one seam.

mod arg_reduce;
mod attention;
mod backend;
mod convolution;
mod cross_entropy;
mod cross_product;
mod decomposition;
mod dense_product;
mod dense_vector;
mod elementwise;
mod pad;
mod parameterized;
mod random;
mod reduction;
mod scan;
mod sparse;
mod stateful;
mod stencil;
mod transfer;
mod volume;
