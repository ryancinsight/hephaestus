use hephaestus_core::Result;

use super::table::require_builtin;
use super::widths::{CanaryWidth, MathBuiltin};
use crate::infrastructure::device::WgpuDevice;

const TOKENS: &[(MathBuiltin, &str)] = &[
    (MathBuiltin::Exp, "exp("),
    (MathBuiltin::Log, "log("),
    (MathBuiltin::Pow, "pow("),
    (MathBuiltin::Tanh, "tanh("),
];

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Scan WGSL source for the canary builtins, honoring identifier boundaries.
///
/// `atanh(` must not match `tanh(`, and `log2(`/`log10(`/`lgamma(` must not
/// match `log(`; the template text is the ground truth of what the driver
/// will compile, so composite expressions are covered without a manual map.
#[must_use]
pub fn scan_builtins(source: &str) -> Vec<MathBuiltin> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    for (builtin, token) in TOKENS {
        let token = token.as_bytes();
        if bytes.len() < token.len() {
            continue;
        }
        let mut matched = false;
        for start in 0..=(bytes.len() - token.len()) {
            if bytes[start..start + token.len()] != *token {
                continue;
            }
            if start > 0 && is_ident_byte(bytes[start - 1]) {
                continue;
            }
            matched = true;
            break;
        }
        if matched {
            found.push(*builtin);
        }
    }
    found
}

/// Refuse dispatch when `source` uses a crash-class builtin at `width`.
///
/// Callers pass the template or composed WGSL they are about to compile; each
/// [`MathBuiltin`] found by [`scan_builtins`] goes through [`require_builtin`](super::require_builtin).
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`](hephaestus_core::HephaestusError::Unsupported) naming the operation and cause for
/// the first blocked builtin found.
pub fn require_expr_safe(
    device: &WgpuDevice,
    source: &str,
    width: CanaryWidth,
    operation: &'static str,
) -> Result<()> {
    for builtin in scan_builtins(source) {
        require_builtin(device, builtin, width, operation)?;
    }
    Ok(())
}
