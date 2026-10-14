use super::*;
use crate::domain::error::HephaestusError;
use crate::test_support::assert_rejects;

// ── operand validators ───────────────────────────────────────────

mod cholesky;
mod fixtures;
mod lu;
mod operands;
mod qr;
mod svd;
pub(crate) use fixtures::*;

// ── panel_lu_packed ──────────────────────────────────────────────

// ── panel_qr_packed ──────────────────────────────────────────────

// ── split_packed_lu ──────────────────────────────────────────────
