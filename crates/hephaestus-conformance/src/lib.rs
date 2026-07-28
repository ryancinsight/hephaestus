#![deny(missing_docs)]
//! # hephaestus-conformance
//!
//! One set of contract clauses that every accelerator backend is held to.
//!
//! Before this crate, each backend carried a hand-written `tests/contract.rs`
//! and the four had diverged: of the 112 entry points declared by all four
//! backends, only 46 were exercised by all four, and six were exercised by none
//! (Atlas conformance triage, 2026-07-28). The contract of a substitution seam
//! was in practice defined by whichever backend's author wrote the most tests.
//!
//! The clauses here are generic over [`ComputeDevice`] and the operation seam,
//! so a backend runs them by instantiating rather than by re-authoring, and a
//! clause added once is executed by every backend from then on.
//!
//! ## Shape
//!
//! Every clause is a free function taking the device and the seam value, and
//! panicking with a located message on violation — the shape a test harness
//! expects, without this crate depending on one. Clauses are grouped by the seam
//! they exercise, one module per seam.
//!
//! ## Oracles
//!
//! Clauses assert exact equality wherever the arithmetic admits it, and derive
//! any tolerance they do need from the operation rather than from observation.
//! The reduction clauses use integer-valued `f32` operands whose products stay
//! below `2^24`, where `f32` represents every intermediate exactly and addition
//! and multiplication are associative over the values involved — so reduction
//! order cannot change the result and no tolerance is applicable. A clause that
//! needed an epsilon would state its derivation at the assertion site.

/// Contract clauses for the [`AxisReductionOps`](hephaestus_core::AxisReductionOps) seam.
pub mod axis_reduction;

pub use axis_reduction::assert_axis_reduction_contract;
