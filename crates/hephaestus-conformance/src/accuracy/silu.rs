//! Accuracy sweep for [`SiluOp`](hephaestus_core::SiluOp).
//!
//! The renderings multiply the stabilized two-sided sigmoid by `x` (the
//! same quotient slice 2 pins for [`SigmoidOp`](hephaestus_core::SigmoidOp),
//! shared string): the `x <= 0` branch evaluates `x·exp(x)/(1+exp(x))`
//! where `exp` cannot overflow, the `x > 0` branch `x/(1+exp(-x))`, and
//! `x = -∞` guards to `-0` (the unguarded form makes `-∞·0`). The sweep
//! covers the `exp` overflow knees past `|x| = ln(FLT_MAX)`, the ADR defect
//! window `[-91.86, -88.72]` where the value stays a normal float, and the
//! structural specials. The negative tail splits at `exp` normality
//! (`x = -87`): above it the intermediate is normal, below it the
//! subnormal intermediate's fixed rounding amplifies by `|x|` against a
//! still-normal result.

use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, SiluOp, UnaryExpr};

use super::{
    SweepDialect, above_flush_floor, assert_unary_sweep_within, sweep_inputs, without_subnormals,
};

/// 4096 points over `[-10, 10]`, 256 over the positive tail, plus the
/// tail-band handoff and the structural specials: everything at or above
/// the negative mid band, swept at the form bound. The positive tail
/// sweeps here because its error vanishes (the result nears `x`).
fn main_inputs() -> Vec<f32> {
    // The builder appends the specials once per call; the duplicate sets are
    // harmless (a repeated input re-verifies the same lane).
    let mut inputs = sweep_inputs(
        -10.0,
        10.0,
        4096,
        &[
            88.73,
            89.0,
            100.0,                       // positive saturation interior
            f32::from_bits(0xc0c0_0000), // exact allowed-exp countermodel (-6)
            -10.0,                       // tail-band handoff (also covered by the mid sweep)
            0.0,                         // +0 (the builder adds -0)
        ],
    );
    inputs.extend(sweep_inputs(10.0, 120.0, 256, &[]));
    inputs
}

/// 512 points over the negative mid band `[-87, -10)`, swept at the
/// driver-`exp` bound: `exp(x)` stays normal, so the error is the builtin
/// grade transferred 1:1 relatively (`x` multiplies exactly-ish) — the
/// same grade the `sigmoid` tail band records.
fn mid_inputs() -> Vec<f32> {
    sweep_inputs(
        -87.0,
        -10.0,
        512,
        &[
            -87.0, // far-band handoff (also covered by the far sweep)
            -20.0, // saturation interior
            -10.0, // tail-band handoff (also covered by the main sweep)
        ],
    )
}

/// 512 points over the negative far band `[-93, -87)`, swept at the
/// subnormal-intermediate envelope: `exp(x)` goes subnormal while `x`
/// keeps the result normal, so the intermediate's fixed `2^-149`
/// rounding amplifies by `|x|` against the result's ULP. Covers the ADR
/// defect window, the `exp` knees, and the deep underflow lanes.
fn far_inputs() -> Vec<f32> {
    sweep_inputs(
        -93.0,
        -87.0,
        512,
        &[
            -87.0,  // far-band handoff (also covered by the mid sweep)
            -88.72, // just above the exp knee
            -88.73, // just past it: exp overflows under the old form
            -89.0,  // saturation interior
            -91.5,  // underflow-transition interior
            -91.86, // the ADR defect-window bottom (CUDA/HIP only)
            -92.0,  // subnormal-result interior (CUDA/HIP only)
            -110.0, // deep underflow lane (CUDA/HIP only)
            -120.0, // clean zero past here (CUDA/HIP only)
        ],
    )
}

/// Sweep [`SiluOp`](hephaestus_core::SiluOp) on one backend.
///
/// Bounds: main CUDA `6` ULP (`expf` at `2`, `/` at `2` without
/// `-prec-div`, products at `0.5`, plus margin); HIP `6` ULP (the CUDA
/// rows carried unconfirmed — any excess on hardware CI is a finding);
/// WGSL `16` ULP (the `sigmoid` main bound plus the `x` product, with
/// the driver-`exp` term recorded at `≈ 6.6` — Metal measures `8.1`
/// total worst at `x = -5.95`, inside the `(3+2|x|)` envelope — plus
/// margin for cross-driver variance). The mid band carries the
/// driver-`exp` grade: CUDA/HIP `8` ULP (the native
/// builtin holds its row over normal intermediates, plus margin), WGSL
/// `64` ULP (the Vulkan large-argument `exp` grade near `52` the
/// `sigmoid` tail band records). The far band carries the
/// subnormal-intermediate envelope: at the ADR bottom `x = -91.86` the
/// worst case is `91.86·2.5·2^-149/2^-149 ≈ 230` (correct rounding plus
/// the `2`-ULP `expf` spec against the bottom-normal-bin ULP), so
/// CUDA/HIP carry `256`; WGSL carries `256` over the surviving
/// handoff/specials, and the bound still catches the hunted defect with
/// four orders of margin (`0` against a normal float scores past `1e7`
/// ULP). WGSL drops `x < -87.0` (matching `sigmoid.rs`): below
/// `exp` normality a flushing driver zeroes the intermediate while `x`
/// keeps the true result normal, corrupting an in-contract result — the
/// floor follows the intermediate, not the result. CUDA and HIP sweep
/// the tail as IEEE.
pub fn assert_silu_sweep<D, E>(device: &D, ops: &E, dialect: SweepDialect)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SiluOp: UnaryExpr<E::Dialect>,
{
    dialect.check::<E::Dialect>();
    let (max_ulp, mid_ulp, far_ulp) = match dialect {
        SweepDialect::CudaC | SweepDialect::HipC => (6.0, 8.0, 256.0),
        SweepDialect::Wgsl => (16.0, 64.0, 256.0),
    };
    let main = main_inputs();
    let mid = mid_inputs();
    let far = far_inputs();
    // WGSL permits flushing subnormals, inputs and results alike; the other
    // dialects sweep them.
    let (main, mid, far) = if dialect == SweepDialect::Wgsl {
        (
            above_flush_floor(&without_subnormals(&main), -87.0),
            above_flush_floor(&without_subnormals(&mid), -87.0),
            above_flush_floor(&without_subnormals(&far), -87.0),
        )
    } else {
        (main, mid, far)
    };
    assert_unary_sweep_within::<D, E, SiluOp>(device, ops, "silu", &main, max_ulp);
    assert_unary_sweep_within::<D, E, SiluOp>(device, ops, "silu-mid", &mid, mid_ulp);
    assert_unary_sweep_within::<D, E, SiluOp>(device, ops, "silu-far", &far, far_ulp);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_sweep_retains_intrinsic_countermodel() {
        let _trace = super::super::runner::trace();
        let _assert = tracing::info_span!(
            "assert",
            op = "silu",
            contract = "exact_countermodel_selection"
        )
        .entered();
        let witness = f32::from_bits(0xc0c0_0000);
        let inputs = above_flush_floor(&without_subnormals(&main_inputs()), -87.0);
        assert!(
            inputs.contains(&witness),
            "exact intrinsic countermodel input removed: {:08x}",
            witness.to_bits()
        );
    }
}
