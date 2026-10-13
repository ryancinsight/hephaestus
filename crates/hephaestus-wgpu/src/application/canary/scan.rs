use hephaestus_core::{DialectScalar, HephaestusError, Result, Wgsl};

use crate::infrastructure::device::WgpuDevice;

use super::verdict::CanaryWidth;
use super::verdict::MathBuiltin;
use super::verdict::require_builtin;
use super::verdict::restricted_width_undefined;

const TOKENS: &[(MathBuiltin, &str)] = &[
    (MathBuiltin::Exp, "exp("),
    (MathBuiltin::Log, "log("),
    (MathBuiltin::Pow, "pow("),
    (MathBuiltin::Tanh, "tanh("),
];

/// The remaining WGSL builtins naga lowers to GLSL.std.450 instructions
/// whose operands the specification restricts to 16 or 32 bits: `Radians`
/// (11) through `Atan2` (25) and `Exp2`/`Log2` (29, 30). They carry no live
/// probe, so they have no [`MathBuiltin`] variant; [`require_expr_safe`]
/// refuses them under the same rule as the probed four.
pub(crate) const RESTRICTED_TOKENS: &[&str] = &[
    "radians(", "degrees(", "sin(", "cos(", "tan(", "asin(", "acos(", "atan(", "sinh(", "cosh(",
    "asinh(", "acosh(", "atanh(", "atan2(", "exp2(", "log2(",
];

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Whether `source` calls `token` (a name followed by `(`) at an identifier
/// boundary, so `atanh(` never matches `tanh(`.
pub(crate) fn calls(source: &[u8], token: &str) -> bool {
    let token = token.as_bytes();
    source
        .windows(token.len())
        .enumerate()
        .any(|(start, window)| window == token && (start == 0 || !is_ident_byte(source[start - 1])))
}

/// Scan WGSL source for the canary builtins, honoring identifier boundaries.
///
/// `atanh(` must not match `tanh(`, and `log2(`/`log10(`/`lgamma(` must not
/// match `log(`; the template text is the ground truth of what the driver
/// will compile, so composite expressions are covered without a manual map.
#[must_use]
pub fn scan_builtins(source: &str) -> Vec<MathBuiltin> {
    TOKENS
        .iter()
        .filter(|(_, token)| calls(source.as_bytes(), token))
        .map(|(builtin, _)| *builtin)
        .collect()
}

/// Refuse dispatch when `source` uses a blocked builtin at `width`.
///
/// Callers pass the template or composed WGSL they are about to compile; each
/// [`MathBuiltin`] found by [`scan_builtins`] goes through [`require_builtin`],
/// and every other GLSL.std.450 precision-restricted builtin is refused under
/// the same backend-and-width rule.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] naming the operation and cause for
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
    if restricted_width_undefined(device, width)
        && let Some(token) = RESTRICTED_TOKENS
            .iter()
            .find(|token| calls(source.as_bytes(), token))
    {
        let name = token.trim_end_matches('(');
        return Err(HephaestusError::Unsupported {
            message: format!(
                "{operation}: WGSL `{name}` at {width:?} is blocked on this backend: naga lowers it to a GLSL.std.450 instruction that admits only 16/32-bit operands"
            ),
        });
    }
    Ok(())
}

/// Canary width of the WGSL type a template is instantiated at, read from its
/// token so vector and composite types count by their component:
/// `vec2<f64>` is as undefined as `f64` under the GLSL.std.450 rule.
pub(crate) fn token_width(token: &str) -> Option<CanaryWidth> {
    if token.contains("f64") {
        Some(CanaryWidth::F64)
    } else if token.contains("f32") {
        Some(CanaryWidth::F32)
    } else {
        None
    }
}

/// Refuse dispatch when the WGSL template `expr`, instantiated at `T`, uses a
/// builtin blocked at `T`'s component width.
///
/// Types without an f32 or f64 component have no canary width and pass
/// through.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] as [`require_expr_safe`] does.
pub(crate) fn require_template_safe<T: DialectScalar<Wgsl>>(
    device: &WgpuDevice,
    expr: &str,
    operation: &'static str,
) -> Result<()> {
    token_width(T::TYPE_TOKEN).map_or(Ok(()), |width| {
        require_expr_safe(device, expr, width, operation)
    })
}
