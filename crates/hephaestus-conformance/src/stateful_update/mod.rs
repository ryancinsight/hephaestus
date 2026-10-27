//! Contract clauses for provider-owned stateful parameter updates.
//!
//! The clause set is a differential against the Leto CPU oracle: every
//! provider rule must land on the same parameter and state values the
//! `leto_ops` reference produces for the same inputs and parameters, to
//! within a tolerance derived from the rule's own rounding budget.
//!
//! The module is split along the seam boundaries rather than kept in one
//! file:
//!
//! - `width` carries the width-indexed dispatch seam that lets one
//!   clause body run for both `f32` and `f64`.
//! - `oracle` carries the Leto CPU reference rules and the seeded
//!   fixture operands they step over.
//! - `harness` carries the machinery that runs a rule on the host and
//!   on a backend and compares the two.
//! - `clauses` carries the clauses themselves: the per-rule
//!   differential, the rank-boundary layout sweep, the atomicity
//!   clause, and the aggregate entry point.

mod clauses;
mod harness;
mod oracle;
mod width;

pub use clauses::assert_stateful_update_contract;
pub use width::StatefulUpdateWidth;
