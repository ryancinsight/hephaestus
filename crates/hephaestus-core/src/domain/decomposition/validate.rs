use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;
use leto::Layout;

/// Validate that a rank-2 operand is square and fits its buffer, returning its
/// dimension `n`.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when the shape is not square or
/// the layout does not fit `buf_len` elements.
pub fn validate_square_operand(layout: &Layout<2>, buf_len: usize) -> Result<usize> {
    let [rows, cols] = layout.shape();
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Decomposition requires a square matrix, got shape [{rows}, {cols}]"),
        });
    }
    layout
        .validate_storage_len(buf_len)
        .map_err(map_layout_err)?;
    Ok(rows)
}

/// Require a dense C-contiguous zero-offset operand for a blocked decomposition.
///
/// The blocked entry points bulk-copy `rows·cols` elements straight from the
/// operand's raw buffer. A transposed, offset, or broadcast (zero-stride)
/// layout would make that raw copy compute from the wrong elements — and for
/// layouts whose validated storage extent is smaller than `rows·cols`
/// (broadcast), read past the allocation. `validate_storage_len` only bounds
/// the layout's own extent, so density is checked explicitly here before any
/// raw copy.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] naming the shape, strides, and
/// offset when the operand is not dense C-contiguous at offset 0.
pub fn require_dense_operand(label: &str, layout: &Layout<2>) -> Result<()> {
    if layout.is_c_contiguous() {
        return Ok(());
    }
    Err(HephaestusError::DispatchFailed {
        message: format!(
            "{label} blocked decomposition requires a dense C-contiguous zero-offset operand; \
             got shape {:?}, strides {:?}, offset {} — materialize the view first (e.g. an \
             identity strided copy into a fresh buffer)",
            layout.shape(),
            layout.strides(),
            layout.offset()
        ),
    })
}

/// Validate that every element of `a` is finite.
///
/// Used by all panel factorisation routines before the main computation loop.
/// Consolidates the identical non-finite-check pattern to a single SSOT.
#[inline]
pub(super) fn validate_finite(a: &[f32], op: &str) -> Result<()> {
    if let Some((idx, value)) = a.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op} failed: entry {idx} is non-finite ({value})"),
        });
    }
    Ok(())
}
