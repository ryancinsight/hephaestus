//! Adapterless evaluation of unsigned 32-bit kernel index arithmetic.
//!
//! Naga's constant evaluator folds a WGSL `const` declaration to a literal
//! and wraps u32 `+`, `-` and `*` modulo 2^32 as the device does, so an
//! overflowing expression evaluates to its wrapped value, not an error.
//! Comparing that value against exact `u64` arithmetic pins the expression
//! text a kernel generator emits without a device. The C-family kernels
//! emit the same text: `/`, `%`, `+` and `min` on unsigned 32-bit operands
//! mean the same in WGSL, CUDA C++ and HIP C++.

use hephaestus_core::BlockWidth;
use hephaestus_core::{CudaC, CumSumOp, c_family_axis_scan_source, ceiling_quotient_source};
use wgpu::naga::{Expression, Literal, front::wgsl::parse_str};

/// Numerators at the u32 boundaries of a ceiling quotient by 16: the empty
/// and single-tile cases, the exact multiples on either side, and the range
/// `[2^32 - 15, 2^32 - 1]` where `(n + 15) / 16` wraps.
pub(crate) const TILE_BOUNDARIES: [u32; 8] =
    [0, 1, 15, 16, 17, u32::MAX - 15, u32::MAX - 14, u32::MAX];

/// Evaluate `expression` after the `const` `bindings`, each a WGSL
/// expression over the bindings before it.
pub(crate) fn evaluate(bindings: &[(&str, String)], expression: &str) -> Result<u32, String> {
    let source: String = bindings
        .iter()
        .map(|(name, value)| format!("const {name}: u32 = {value};\n"))
        .chain(core::iter::once(format!(
            "const evaluated: u32 = {expression};\n"
        )))
        .collect();
    let module = parse_str(&source).map_err(|error| error.emit_to_string(&source))?;
    let (_, constant) = module
        .constants
        .iter()
        .find(|(_, constant)| constant.name.as_deref() == Some("evaluated"))
        .ok_or_else(|| format!("no `evaluated` constant in\n{source}"))?;
    match module.global_expressions[constant.init] {
        Expression::Literal(Literal::U32(value)) => Ok(value),
        ref other => Err(format!("`evaluated` folded to {other:?}")),
    }
}

/// The expression a kernel source assigns after `binding = `.
pub(crate) fn assigned_expression<'s>(source: &'s str, binding: &str) -> &'s str {
    let pattern = format!("{binding} = ");
    let mut sites = source.match_indices(&pattern);
    let (start, _) = sites
        .next()
        .expect("invariant: the kernel assigns the binding");
    assert_eq!(sites.next(), None, "`{binding}` is assigned once");
    let rest = &source[start + pattern.len()..];
    &rest[..rest
        .find(';')
        .expect("invariant: the assignment ends with `;`")]
}

/// Assert that `binding`'s expression in `source` equals `⌈n / 16⌉` for
/// every `n` in [`TILE_BOUNDARIES`], with `numerator` bound to `n`.
pub(crate) fn assert_tile_count_is_exact(source: &str, binding: &str, numerator: &str) {
    let expression = assigned_expression(source, binding);
    for n in TILE_BOUNDARIES {
        let expected = u64::from(n).div_ceil(16);
        let got = evaluate(&[(numerator, format!("{n}u"))], expression);
        assert_eq!(
            got.map(u64::from),
            Ok(expected),
            "`{binding} = {expression}` at {numerator} = {n}"
        );
    }
}

#[test]
fn ceiling_quotient_is_exact_at_u32_boundaries() {
    for divisor in [1u32, 2, 16, 256, u32::MAX] {
        let expression = ceiling_quotient_source("n", &format!("{divisor}u"));
        for n in [
            0,
            1,
            divisor - 1,
            divisor,
            divisor.saturating_add(1),
            u32::MAX - 1,
            u32::MAX,
        ] {
            let expected = u64::from(n).div_ceil(u64::from(divisor));
            let got = evaluate(&[("n", format!("{n}u"))], &expression);
            assert_eq!(got.map(u64::from), Ok(expected), "⌈{n} / {divisor}⌉");
        }
    }
}

#[test]
fn scan_chunks_partition_the_line_at_u32_boundaries() {
    for width in [8u32, 256] {
        let block = BlockWidth::new(width).expect("non-zero test width");
        let source = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(block);
        let chunk_len = assigned_expression(&source, "unsigned int chunk_len");
        let start = assigned_expression(&source, "unsigned int start");
        let end = assigned_expression(&source, "unsigned int end");
        for len in [
            0,
            1,
            width - 1,
            width,
            width + 1,
            u32::MAX - width,
            u32::MAX,
        ] {
            let mut covered = 0u64;
            for lane in 0..width {
                let bindings = [
                    ("len", format!("{len}u")),
                    ("lane", format!("{lane}u")),
                    ("chunk_len", chunk_len.to_string()),
                    ("start", start.to_string()),
                ];
                let start = evaluate(&bindings, "start").map(u64::from);
                let stop = evaluate(&bindings, end).map(u64::from);
                // Lanes cover consecutive, disjoint ranges: a lane starting
                // inside the line starts where the previous one stopped, and
                // a lane starting at or past its end runs no iteration.
                let (start, stop) = match (start, stop) {
                    (Ok(start), Ok(stop)) => (start, stop),
                    failed => panic!("width {width}, len {len}, lane {lane}: {failed:?}"),
                };
                if start < u64::from(len) {
                    assert_eq!(start, covered, "width {width}, len {len}, lane {lane}");
                    covered = stop;
                } else {
                    assert!(stop <= start, "width {width}, len {len}, lane {lane}");
                }
            }
            assert_eq!(covered, u64::from(len), "width {width}, len {len}");
        }
    }
}
