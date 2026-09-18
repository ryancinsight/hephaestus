//! One integration-test binary for the host crate: each module instantiates
//! a shared conformance clause, or pins a host-specific case, for one seam.

mod decomposition;
mod dense_product;
mod dense_vector;
mod elementwise;
mod random;
mod reduction;
mod scan;
mod sparse;
mod transfer;
mod volume;
