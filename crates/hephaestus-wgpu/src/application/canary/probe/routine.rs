//! Width-valid binary64 `exp` built only from operations that GLSL.std.450
//! admits at 64-bit operands: multiply, add, `fma`, `floor`, and `ldexp`.
//!
//! Native `exp` has no defined binary64 result on Vulkan (ADR 0072).
//!
//! # Assumptions
//!
//! WGSL and Vulkan guarantee less than the bound below needs: `fma` may
//! expand to a rounded multiply and add, overflow, infinity and NaN may be
//! assumed absent, subnormals may be flushed, and operations may be
//! reassociated (WGSL "Differences from IEEE-754", "Reassociation and
//! Fusion"; Vulkan "Precision and Operation of SPIR-V Instructions",
//! "Evaluation of Expressions", where only `NoContraction` prevents it and
//! naga emits none); double precision need only match single. The bound holds
//! on an adapter with (1) binary64 add, multiply and `ldexp` correctly
//! rounded to nearest, overflowing to infinity and underflowing gradually,
//! and `floor`, comparisons and `i32` of an integral value exact; (2) NaN,
//! infinity and subnormal operands preserved; (3) operations evaluated as
//! written, with no reassociation or folding of the reduction split (`x +
//! (-k HI - k LO)` gives 508u in a CPU model); `fma` may be fused or not. No
//! WGSL form is shown immune to rearrangement. The sweep test is the
//! evidence for 1-3; the 8-input probe, `|k| <= 29`, is a smoke check.
//!
//! # Algorithm
//!
//! With `k = floor(x log2(e) + 1/2)` and `r = x - k ln 2`, `exp(x) = 2^k
//! exp(r)`. The roundings in `x log2(e) + 1/2`, `log2(e)` included, move it
//! by less than `2^-41`, so `|r| <= a = ln(2)/2 + 2^-41` for `|x| <= 746`. Then:
//!
//! 1. `r = fma(-k, LN2_LO, fma(-k, LN2_HI, x))` with `LN2_HI =
//!    0x1.62e42fee00000p-1` (32 significant bits) and `LN2_LO =
//!    0x1.a39ef35793c76p-33`, the nearest binary64 to `ln 2 - LN2_HI`.
//!    `k LN2_HI` is an exact plain product for `|k| <= 2^21` (`32 + 21 = 53`
//!    bits) and `x - k LN2_HI` is exact (Sterbenz: `x / (k LN2_HI)` lies in
//!    `[1/2, 2]`), so fused or not only the last step rounds, by `u |r|`.
//!    The constants' miss of `ln 2` (`< 2^-86`, times `|k| <= 1076`) and,
//!    unfused, the rounding of `k LN2_LO` add `< 2^-74`.
//! 2. `exp(r)` is the degree-13 Taylor polynomial in Horner form. Its
//!    truncation error relative to `e^r` is at most `0.06 u` (`u = 2^-53`).
//! 3. The result is `ldexp(p, k/2) * ldexp(1, k - k/2)`: the first factor is
//!    exact and the product is one IEEE multiplication, so under assumption 1
//!    overflow, gradual underflow and rounding are defined without `ldexp`
//!    outside its specified range.
//!
//! # Error bound
//!
//! Relative to `e^r`, worst at `r = -a` where `e^-a = 1/sqrt 2`. Reduction:
//! `u |r| + 2^-74 <= 0.35 u`. Polynomial: the coefficient of `r^m` carries
//! `2m + 1` roundings unfused (`m + 1` fused), so to first order the error
//! is `u sum_m (2m + 1) |r|^m / m!` (fused `(m + 1)`), plus `u (e^a - 1 - a -
//! a^2/2)` for the rounded coefficients `m >= 3`. The sums are `e^a (1 + 2a)`
//! and `e^a (1 + a)`, giving `2 (1 + ln 2) + 0.011 = 3.40 u` unfused and
//! `2 + ln 2 + 0.011 = 2.70 u` fused. With truncation the totals are
//! `3.81 u` and `3.11 u`; second-order terms are a relative `27 u`. Stated as
//! `RELATIVE_ERROR_BOUND = 4 u` for normal results; a subnormal result adds
//! half its spacing, `2^-1075`.
//!
//! # Special values
//!
//! Under assumptions 1 and 2, NaN returns NaN; `x > 710` returns `+Inf`
//! (`exp(710) > DBL_MAX`); `x < -746` returns `+0` (`exp(-746) < 2^-1075`
//! rounds to zero), so `+-Inf` fall in these cases; `+-0` gives `1`.

use hephaestus_core::{BlockWidth, ComputeDevice, Result};

use crate::application::elementwise::encode_elementwise;
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::infrastructure::device::WgpuDevice;

/// Relative error bound `4 u`, `u = 2^-53`, of a normal result.
#[cfg(test)]
pub(super) const RELATIVE_ERROR_BOUND: f64 = 4.0 * (f64::EPSILON / 2.0);

struct ExpRoutine;

/// WGSL `exp_routine(x: f64) -> f64`; the Taylor coefficients are `1/j!`
/// rounded to nearest.
const ROUTINE: &str = "
fn exp_routine(x: f64) -> f64 {
    if (x != x) { return x; }
    if (x > 710.0) { return x * 1.0e308; }
    if (x < -746.0) { return 0.0; }
    let k = floor(fma(x, 1.4426950408889634, 0.5));
    let r = fma(-k, 1.9082149292705877e-10, fma(-k, 0.6931471803691238, x));
    var p = f64(1.6059043836821613e-10);
    p = fma(p, r, 2.08767569878681e-09);
    p = fma(p, r, 2.505210838544172e-08);
    p = fma(p, r, 2.755731922398589e-07);
    p = fma(p, r, 2.7557319223985893e-06);
    p = fma(p, r, 2.48015873015873e-05);
    p = fma(p, r, 0.0001984126984126984);
    p = fma(p, r, 0.001388888888888889);
    p = fma(p, r, 0.008333333333333333);
    p = fma(p, r, 0.041666666666666664);
    p = fma(p, r, 0.16666666666666666);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    let n = i32(k);
    let half = n / 2;
    return ldexp(p, half) * ldexp(f64(1.0), n - half);
}
";

fn source() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read> input: array<f64>;\n@group(0) @binding(1) var<storage, read_write> out: array<f64>;\n{ROUTINE}\n@compute @workgroup_size({width})\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let i = id.x;\n    if (i >= arrayLength(&input)) {{ return; }}\n    out[i] = exp_routine(input[i]);\n}}",
        width = BlockWidth::DEFAULT.get(),
    )
}

/// Evaluate the routine over `inputs` in one dispatch.
///
/// # Errors
///
/// Surfaces allocation, pipeline-compilation (a device without `ShaderF64`
/// fails validation), dispatch, and transfer failures.
pub(super) fn evaluate(device: &WgpuDevice, inputs: &[f64]) -> Result<Vec<f64>> {
    let input = device.upload(inputs)?;
    let output = device.alloc_uninitialized::<f64>(inputs.len())?;
    let entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: input.buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: output.buffer.as_entire_binding(),
        },
    ];
    let pipeline = try_cached_pipeline(
        device,
        (
            std::any::TypeId::of::<ExpRoutine>(),
            std::any::TypeId::of::<f64>(),
            0,
        ),
        "hephaestus-exp-routine",
        source,
    )?;
    let groups = workgroups(inputs.len(), BlockWidth::DEFAULT)?;
    encode_elementwise(
        device,
        &pipeline,
        "hephaestus-exp-routine",
        &entries,
        groups,
    )?;
    device.download_owned(&output)
}

#[cfg(test)]
mod tests;
