//! Contract clauses for the device-neutral [`hephaestus_core::SortOps`] seam.
//!
//! There is no `leto::sort` to delegate to, so the host reference carries
//! the direct implementation and these clauses pin its value semantics
//! with hand-computed fixtures: stably sort along one axis into the
//! sorted values plus the `u32` argsort positions — the `torch.sort`
//! semantic `coeus_ops::sort` documents, except the indices are `u32`
//! rather than `T`-encoded. Fixtures use small signed integers so every
//! comparison is an exact equality, never a tolerance-bounded one.
//!
//! Stability is pinned observably through the indices: equal values keep
//! their input order in both directions, which a `(value, position)`
//! total order guarantees but an unstable network/net cannot. `NaN`
//! sorts as greater than every other value — last ascending, first
//! descending — except where the dialect opts out: on a non-advertising
//! dialect an unordered `NaN` comparison can corrupt the whole lane's
//! order, so the `NaN`-mixed fixtures skip wholesale by the ADR 0043
//! const branch, while the `-0.0`/`-inf` fixtures assert on every
//! backend.

mod guards;
mod special;
mod values;

pub use guards::{
    assert_sort_contract, assert_sort_of_empty_is_vacuous, assert_sort_rejects_aliased_output,
    assert_sort_rejects_invalid_geometry,
};
pub use special::assert_sort_special_float_values;
pub use values::{assert_sort_stability, assert_sort_value_contract};
