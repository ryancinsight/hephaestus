//! Leto view plumbing shared by the three prepared forms and, crate-wide,
//! by the runtime-parameter and stateful-update seams: broadcast, output
//! validation, the shared layout diagnostic, and the write step.

use hephaestus_core::{HephaestusError, Result};
use leto::{ElementIterMut, Layout};

use crate::HostBuffer;
use crate::combine::unsupported_operator;

/// Map a leto layout error the same way `hephaestus-wgpu`'s
/// `application::strided::map_layout_err` does: the shared elementwise
/// conformance clauses (`hephaestus-conformance`) assert this exact
/// `"layout rejected: "` wording for a rejected broadcast, so every backend's
/// diagnostic names the violated constraint identically.
///
/// `pub(crate)`: shared verbatim with [`crate::parameterized`], whose
/// runtime-parameter unary seam broadcasts and iterates the same way.
pub(crate) fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

/// Broadcast `layout` to `target_shape` and validate it against `storage_len`.
/// Mirrors `hephaestus-wgpu`'s per-operand broadcast + `validate_storage_len`
/// step in `prepare_*_inner`.
///
/// `pub(crate)`: shared with [`crate::parameterized`] (see [`map_layout_err`]).
pub(crate) fn broadcast_operand<const N: usize>(
    layout: &Layout<N>,
    target_shape: [usize; N],
    storage_len: usize,
) -> Result<Layout<N>> {
    let broadcast = layout.broadcast(target_shape).map_err(map_layout_err)?;
    broadcast
        .validate_storage_len(storage_len)
        .map_err(map_layout_err)?;
    Ok(broadcast)
}

/// Validate an elementwise output layout: matches its buffer's storage and is
/// non-overlapping (no two logical positions writing the same physical
/// element). Mirrors `hephaestus-wgpu`'s `validate_out`.
pub(super) fn validate_elementwise_output<T, const N: usize>(
    output: &HostBuffer<T>,
    out_layout: &Layout<N>,
) -> Result<()> {
    out_layout
        .validate_storage_len(output.read().len())
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must be non-overlapping".to_string(),
        });
    }
    Ok(())
}

/// Write the next logical output element from an applied value, or the typed
/// dispatch failure when the operator reported no application.
///
/// `pub(crate)`: shared with [`crate::parameterized`] (see [`map_layout_err`]).
pub(crate) fn write_next<T, const N: usize>(
    out_iter: &mut ElementIterMut<'_, T, N>,
    applied: Option<T>,
    op_name: &str,
) -> Result<()> {
    let slot = out_iter
        .next()
        .expect("invariant: broadcast operands and output share element count");
    *slot = applied.ok_or_else(|| unsupported_operator(op_name))?;
    Ok(())
}
