//! Operand access shared by the host seam implementors.
//!
//! Every [`HostBuffer`] is one `RwLock`, so an operation naming one buffer in
//! two roles must take a single guard for both: a second guard on a lock the
//! thread already holds is a deadlock or a panic, never an alias.

use std::sync::RwLockReadGuard;

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

/// Read an arbitrary number of read-only operand buffers, taking exactly one
/// guard per distinct underlying allocation.
///
/// Attention and convolution admit read-operand lists (query/key/value/keep
/// mask; input/weight/bias) with no restriction against the operands aliasing
/// each other — only a writable destination is ever rejected for aliasing a
/// read operand — so a dispatch naming one buffer under two or more roles
/// (self-attention's `query == key == value`, for example) must still take
/// exactly one guard for it, per [`with_operands`]'s reentrant-lock hazard.
/// `body` receives one slice per entry of `buffers`, in the same order,
/// backed by the deduplicated guards.
pub(crate) fn with_operand_reads<T, R>(
    buffers: &[&HostBuffer<T>],
    body: impl FnOnce(&[&[T]]) -> R,
) -> R {
    let mut guards: Vec<RwLockReadGuard<'_, Vec<T>>> = Vec::with_capacity(buffers.len());
    let mut owners: Vec<usize> = Vec::with_capacity(buffers.len());
    for (index, buffer) in buffers.iter().enumerate() {
        let owner = buffers[..index]
            .iter()
            .position(|other| other.aliases(buffer))
            .map_or_else(
                || {
                    guards.push(buffer.read());
                    guards.len() - 1
                },
                |previous| owners[previous],
            );
        owners.push(owner);
    }
    let slices: Vec<&[T]> = owners
        .iter()
        .map(|&owner| guards[owner].as_slice())
        .collect();
    body(&slices)
}
