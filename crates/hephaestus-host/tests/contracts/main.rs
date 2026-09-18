//! One integration-test binary for the host crate: each module instantiates
//! a shared conformance clause, or pins a host-specific case, for one seam.

mod decomposition;
mod dense_product;
mod dense_vector;
mod random;
mod sparse;
mod transfer;
