//! Operand access shared by the host seam implementors.
//!
//! Every [`HostBuffer`] is one `RwLock`, so an operation naming one buffer in
//! two roles must take a single guard for both: a second guard on a lock the
//! thread already holds is a deadlock or a panic, never an alias.

use hephaestus_core::{HephaestusError, Result};

use crate::HostBuffer;

/// Reject an output buffer that is the same underlying allocation as either
/// input.
///
/// The host represents every buffer as `Arc<RwLock<Vec<T>>>`
/// ([`HostBuffer`]); reading an operand and writing the output through the
/// same lock would deadlock rather than alias silently, so this check turns
/// that hang into the typed error the seam contract already documents.
pub(crate) fn require_disjoint_output<T>(
    lhs: &HostBuffer<T>,
    rhs: &HostBuffer<T>,
    output: &HostBuffer<T>,
) -> Result<()> {
    if lhs.aliases(output) || rhs.aliases(output) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }
    Ok(())
}

/// Read both operands, under one guard when they are the same allocation.
///
/// `std::sync::RwLock::read` documents that it may panic when the current
/// thread already holds the lock, and a call naming one buffer twice —
/// squaring by `matmul(a, a)`, `dot(a, a)` — would take exactly that second
/// guard.
pub(crate) fn with_operands<T, R>(
    lhs: &HostBuffer<T>,
    rhs: &HostBuffer<T>,
    body: impl FnOnce(&[T], &[T]) -> R,
) -> R {
    let lhs_cells = lhs.read();
    if rhs.aliases(lhs) {
        return body(&lhs_cells, &lhs_cells);
    }
    let rhs_cells = rhs.read();
    body(&lhs_cells, &rhs_cells)
}
