//! Width-valid binary64 `exp` built only from operations that GLSL.std.450
//! admits at 64-bit operands: multiply, add, `fma`, `floor`, and `ldexp`.
//!
//! Native `exp` has no defined binary64 result on Vulkan (ADR 0072); this is
//! the replacement the ADR names. Seam adoption is a separate item.
//!
//! # Algorithm
//!
//! With `k = floor(x log2(e) + 1/2)` and `r = x - k ln 2`, `exp(x) = 2^k
//! exp(r)` and `|r| <= ln(2)/2 + delta`, where `delta < 2^-42` is the
//! rounding of `x log2(e)` at `|x| <= 746`. Then:
//!
//! 1. `r` is formed by two fused steps, `fma(-k, LN2_LO, fma(-k, LN2_HI, x))`,
//!    with `LN2_HI` the nearest binary64 to `ln 2` and `LN2_LO` the nearest to
//!    `ln 2 - LN2_HI`. Each `fma` rounds once; the constant error
//!    `k |ln 2 - LN2_HI - LN2_LO| < 2^-98` is negligible.
//! 2. `exp(r)` is the degree-13 Taylor polynomial in Horner form with fused
//!    steps. Its truncation error is `|r|^14 e^|r| / 14! < 5.9e-18 = 0.06 u`
//!    (`u = 2^-53`).
//! 3. The result is `ldexp(p, k/2) * ldexp(1, k - k/2)`. The first factor is
//!    exact; the product is a single IEEE multiplication, so overflow,
//!    gradual underflow and the final rounding are defined without relying on
//!    `ldexp` outside its specified range.
//!
//! # Error bound
//!
//! Each of the 13 fused Horner steps and each of the 12 coefficient roundings
//! contributes at most `u` of its partial value (`<= e^0.3466 = 1.414`)
//! weighted by `|r|^j`; the weights sum to `<= 1/(1 - 0.3466) = 1.53`, so
//! the polynomial adds `<= 2 * 1.414 * 1.53 u = 4.33 u`. The two reduction
//! steps add `<= 2 * 0.3466 u = 0.7 u` and truncation `0.06 u`. The total is
//! `5.1 u`, stated as [`RELATIVE_ERROR_BOUND`] `= 6 u` for normal results. A
//! subnormal result carries one further rounding of half the subnormal
//! spacing, `2^-1075`.
//!
//! # Special values
//!
//! NaN returns NaN; `x > 710` returns `+Inf` (`exp(710) > DBL_MAX`); `x < -746`
//! returns `+0` (`exp(-746) < 2^-1075`, which rounds to zero), so `+-Inf` fall
//! in these cases; `x = +-0` gives `1`.

use hephaestus_core::{BlockWidth, ComputeDevice, Result};

use crate::application::elementwise::encode_elementwise;
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::infrastructure::device::WgpuDevice;

/// Upper bound on the relative error of a normal result: `6 u`, `u = 2^-53`
/// (see the module derivation).
#[cfg(test)]
pub(super) const RELATIVE_ERROR_BOUND: f64 = 6.0 * (f64::EPSILON / 2.0);

struct ExpRoutine;

/// WGSL function `exp_routine(x: f64) -> f64`; `LN2_HI`/`LN2_LO` are
/// `0x1.62e42fefa39efp-1` and `0x1.abc9e3b39803fp-56`, the Taylor
/// coefficients are `1/j!` rounded to nearest.
const ROUTINE: &str = "
fn exp_routine(x: f64) -> f64 {
    if (x != x) { return x; }
    if (x > 710.0) { return x * 1.0e308; }
    if (x < -746.0) { return 0.0; }
    let k = floor(fma(x, 1.4426950408889634, 0.5));
    let r = fma(-k, 2.3190468138462996e-17, fma(-k, 0.6931471805599453, x));
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
