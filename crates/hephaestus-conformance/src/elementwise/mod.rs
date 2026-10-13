//! Contract clauses for untyped unary and binary elementwise dispatch.
//!
//! Complements [`crate::typed_elementwise`], which owns the scalar-aware
//! comparison paths: these clauses cover the plain arithmetic surface —
//! `unary_into`, `binary_into`, their prepared forms, and strided traversal.
//!
//! Every fixture is dyadic (representable as `m·2^e`) and every operation
//! either preserves dyadic values exactly (negation, absolute value,
//! addition, subtraction, multiplication, division by powers of two) or is
//! IEEE-754 correctly rounded onto an exactly representable result
//! (`sqrt` of a perfect square). All oracles are therefore exact
//! equalities; no tolerance is involved.

mod core;
mod gated;
mod special;

pub use core::assert_elementwise_contract;
pub use gated::{
    assert_elu_contract, assert_gelu_contract, assert_grad_tails_contract, assert_mish_contract,
    assert_softplus_contract,
};
pub use special::{
    assert_bessel_contract, assert_expm1_log1p_contract, assert_k0_contract, assert_sinc_contract,
};
