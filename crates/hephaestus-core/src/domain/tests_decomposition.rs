use super::*;
use crate::domain::error::HephaestusError;
use crate::test_support::assert_rejects;

// ── operand validators ───────────────────────────────────────────

#[path = "decomposition_tests/cholesky.rs"]
mod cholesky;
#[path = "decomposition_tests/fixtures.rs"]
mod fixtures;
#[path = "decomposition_tests/lu.rs"]
mod lu;
#[path = "decomposition_tests/operands.rs"]
mod operands;
#[path = "decomposition_tests/qr.rs"]
mod qr;
#[path = "decomposition_tests/svd.rs"]
mod svd;
pub(crate) use fixtures::*;

// ── panel_lu_packed ──────────────────────────────────────────────

// ── panel_qr_packed ──────────────────────────────────────────────

// ── split_packed_lu ──────────────────────────────────────────────
