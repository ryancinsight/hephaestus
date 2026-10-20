//! Stabilized C-dialect expressions for the tail-sensitive activations.
//!
//! Ported from the HEPH-KERNEL-TAIL-ACCURACY rescue work (ADR 0061
//! Decisions 6/7): the retired one-sided C spellings overflowed `exp` on
//! the tails (`sigmoid(-88.73)` rendered `0` instead of a subnormal) and
//! formed `inf * 0` NaNs at the infinities. Each composer parents its
//! output, so assemblies splice it under `*`, `/`, comparisons, and
//! ternary conditions alike. The C assemblies serve both C dialects:
//! CUDA verifies them on-device, and the string pins record the shared
//! spelling until HIP hardware says otherwise.
//!
//! The WGSL GELU-tanh gradient also bounds its eagerly evaluated operands
//! before polynomial evaluation and uses a nonoverflowing sigmoid product.

use super::super::UnaryExpr;
use super::super::{GeluTanhGradOp, GeluTanhOp, MishGradOp, MishOp, SiluGradOp, SiluOp};
use crate::domain::dialect::{CudaC, HipC, Wgsl};

// Branch-selected sigmoid over the literal argument `w`: the `w <= 0`
// half evaluates `exp(w)` where it cannot overflow, the `w > 0` half
// `exp(-w)`.
macro_rules! c_stable_sigmoid {
    ($w:literal) => {
        concat!(
            "((",
            $w,
            ") <= 0.0f ? exp((",
            $w,
            ")) / (1.0f + exp((",
            $w,
            "))) : 1.0f / (1.0f + exp(-(",
            $w,
            "))))"
        )
    };
}

// The `SoftplusOp` C spelling, parenthesized so the mish assemblies splice
// it under `tanh`, `*`, and the `sp` crossover comparison.
macro_rules! c_softplus {
    () => {
        "((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))"
    };
}

// `w = 2z`, `z = c0·(x + c1·x³)`, spelled twice from one source: both halves
// evaluate identically, and doubling is exact short of overflow, matching
// the value function (`z + z`) bit for bit including the overflow to `±inf`.
macro_rules! c_gelu_tanh_w {
    () => {
        "((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))"
    };
}

// Fixed-`w` twin of the sigmoid composer: `macro_rules!` `:literal`
// cannot carry a macro expansion, so `w` splices through `concat!`.
macro_rules! c_gelu_tanh_s {
    () => {
        concat!(
            "((",
            c_gelu_tanh_w!(),
            ") <= 0.0f ? exp((",
            c_gelu_tanh_w!(),
            ")) / (1.0f + exp((",
            c_gelu_tanh_w!(),
            "))) : 1.0f / (1.0f + exp(-(",
            c_gelu_tanh_w!(),
            "))))"
        )
    };
}

// The independent saturation `s·σ(−w)`, pre-grouped like the value
// function. `σ(−w)` spells its negative half directly: this composer only
// splices past the `w > 2.47` crossover, where `−w < 0` always, so the
// positive half is unreachable and the direct half is exactly what the
// branch-selected sigmoid computes there (NaN included: both spellings
// propagate it through a failed comparison).
macro_rules! c_gelu_tanh_sat_indep {
    () => {
        concat!(
            "(",
            c_gelu_tanh_s!(),
            " * (exp(-(",
            c_gelu_tanh_w!(),
            ")) / (1.0f + exp(-(",
            c_gelu_tanh_w!(),
            ")))))"
        )
    };
}

// `sech²(sp) = 4u/(1+u)²`, `u = exp(−2·sp)`, for the mish-grad crossover.
macro_rules! c_mish_sech2 {
    () => {
        concat!(
            "((4.0f * exp(-2.0f * ",
            c_softplus!(),
            ")) / ((1.0f + exp(-2.0f * ",
            c_softplus!(),
            ")) * (1.0f + exp(-2.0f * ",
            c_softplus!(),
            "))))"
        )
    };
}

macro_rules! c_silu_expr {
    () => {
        concat!(
            "(x) < -3.402823466e+38f ? -0.0f : (x) * ",
            c_stable_sigmoid!("(x)")
        )
    };
}

macro_rules! c_silu_grad_expr {
    () => {
        concat!(
            "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : ",
            c_stable_sigmoid!("(x)"),
            " * (1.0f + (x) * ((x) <= 2.63f ? 1.0f - ",
            c_stable_sigmoid!("(x)"),
            " : ",
            c_stable_sigmoid!("(-(x))"),
            ")))"
        )
    };
}

macro_rules! c_mish_expr {
    () => {
        concat!(
            "(x) < -3.402823466e+38f ? -0.0f : (x) * tanh(",
            c_softplus!(),
            ")"
        )
    };
}

macro_rules! c_mish_grad_expr {
    () => {
        concat!(
            "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : tanh(",
            c_softplus!(),
            ") + (x) * ((",
            c_softplus!(),
            ") <= 1.55f ? 1.0f - tanh(",
            c_softplus!(),
            ") * tanh(",
            c_softplus!(),
            ") : ",
            c_mish_sech2!(),
            ") * ",
            c_stable_sigmoid!("(x)"),
            ")"
        )
    };
}

macro_rules! c_gelu_tanh_expr {
    () => {
        concat!("(x) < -3.402823466e+38f ? -0.0f : (x) * ", c_gelu_tanh_s!())
    };
}

macro_rules! c_gelu_tanh_grad_expr {
    () => {
        concat!(
            "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : ((",
            c_gelu_tanh_w!(),
            ") <= 2.47f ? (",
            c_gelu_tanh_s!(),
            " * (1.0f - ",
            c_gelu_tanh_s!(),
            ") == 0.0f ? ",
            c_gelu_tanh_s!(),
            " : ",
            c_gelu_tanh_s!(),
            " + 2.0f * (x) * ",
            c_gelu_tanh_s!(),
            " * (1.0f - ",
            c_gelu_tanh_s!(),
            ") * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x))) : (",
            c_gelu_tanh_sat_indep!(),
            " == 0.0f ? ",
            c_gelu_tanh_s!(),
            " : ",
            c_gelu_tanh_s!(),
            " + 2.0f * (x) * ",
            c_gelu_tanh_sat_indep!(),
            " * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x)))))"
        )
    };
}

// `x·sigmoid(x)` through the branch-selected sigmoid (ADR 0061 Decision 6):
// the retired `x / (1 + exp(-x))` overflowed `exp` past `x = -88.73` and made
// `-inf/inf` at `x = -inf`. The guard catches only `-inf` (finite inputs
// stay past `-FLT_MAX`); `+inf` needs none (`inf·1`), and NaN propagates
// through the failed comparisons.
impl UnaryExpr<Wgsl> for SiluOp {
    const EXPR: &'static str = "select(x / (1.0 + exp(-x)), -0.0, (x) < -3.402823466e+38)";
}
impl UnaryExpr<CudaC> for SiluOp {
    const EXPR: &'static str = c_silu_expr!();
}
impl UnaryExpr<HipC> for SiluOp {
    const EXPR: &'static str = c_silu_expr!();
}

// `s·(1 + x·(1−s))`, taking the input per ADR 0061 Decision 7. `1 − s` goes
// direct below `x = 2.63` and independent above (`SILU_GRAD_CROSSOVER`).
// Both infinities guard explicitly (`±inf·0` unguarded); NaN propagates.
impl UnaryExpr<Wgsl> for SiluGradOp {
    const EXPR: &'static str = "select(select((1.0 / (1.0 + exp(-x))) * (1.0 + x * (1.0 - (1.0 / (1.0 + exp(-x))))), 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)";
}
impl UnaryExpr<CudaC> for SiluGradOp {
    const EXPR: &'static str = c_silu_grad_expr!();
}
impl UnaryExpr<HipC> for SiluGradOp {
    const EXPR: &'static str = c_silu_grad_expr!();
}

// `x·tanh(softplus(x))`, composing the two-sided softplus (ADR 0061
// Decision 6): the retired `log(1 + exp(x))` overflowed `exp` past
// `x = 88.73` and rounded `1 + exp(x)` to `1` below `x = -16.6`, zeroing the
// whole negative tail. Same `-inf` guard as SiLU.
impl UnaryExpr<CudaC> for MishOp {
    const EXPR: &'static str = c_mish_expr!();
}
impl UnaryExpr<HipC> for MishOp {
    const EXPR: &'static str = c_mish_expr!();
}

// `t + x·(1−t²)·σ(x)` with `t = tanh(sp)`, taking the input per ADR 0061
// Decision 7. `1 − t²` goes direct below `sp = 1.55` and `sech²` above
// (`MISH_GRAD_CROSSOVER`, compared on `sp`, not `t`); both infinities guard
// explicitly.
impl UnaryExpr<CudaC> for MishGradOp {
    const EXPR: &'static str = c_mish_grad_expr!();
}
impl UnaryExpr<HipC> for MishGradOp {
    const EXPR: &'static str = c_mish_grad_expr!();
}

// `x·σ(2z)` via `0.5·(1 + tanh(z)) = σ(2z)` (ADR 0061 Decision 6): the
// retired `1 + tanh(z)` cancelled past `z = -4.4`. Same `-inf` guard as SiLU.
impl UnaryExpr<Wgsl> for GeluTanhOp {
    const EXPR: &'static str = "select(x / (1.0 + exp(-2.0 * (0.7978845608 * (x + 0.044715 * x * x * x)))), -0.0, (x) < -3.402823466e+38)";
}
impl UnaryExpr<CudaC> for GeluTanhOp {
    const EXPR: &'static str = c_gelu_tanh_expr!();
}
impl UnaryExpr<HipC> for GeluTanhOp {
    const EXPR: &'static str = c_gelu_tanh_expr!();
}

// WGSL select evaluates both arms. Bound every polynomial operand first:
// q=clamp(x,-40,40), |w|<4632, q²<=1600. With e=exp(-|w|), e is in [0,1]
// and s(1-s)=e/(1+e)² uses neither an overflowing exp nor cancellation.
// Every intermediate stays finite for finite x, including discarded arms.
// For |x|>=40, the exact derivative differs from its 0/1 limit by less than
// 14000*exp(-4600)<1e-1990. The envelope decreases farther out because its
// cubic exponential dominates the degree-three prefactor. This is below half
// a binary64 subnormal ULP, so the same saturation is valid for both lane
// widths. The outer selects enforce those exact limits only after making
// their eagerly evaluated alternative safe. C/HIP retain their lazy arms.
macro_rules! wgsl_gelu_tanh_grad_x {
    () => {
        "clamp(x, -40.0, 40.0)"
    };
}

macro_rules! wgsl_gelu_tanh_grad_e {
    () => {
        concat!(
            "exp(-abs(2.0 * (0.7978845608 * (",
            wgsl_gelu_tanh_grad_x!(),
            " + 0.044715 * ",
            wgsl_gelu_tanh_grad_x!(),
            " * ",
            wgsl_gelu_tanh_grad_x!(),
            " * ",
            wgsl_gelu_tanh_grad_x!(),
            "))))"
        )
    };
}

macro_rules! wgsl_gelu_tanh_grad_s {
    () => {
        concat!(
            "select((",
            wgsl_gelu_tanh_grad_e!(),
            ") / (1.0 + ",
            wgsl_gelu_tanh_grad_e!(),
            "), 1.0 / (1.0 + ",
            wgsl_gelu_tanh_grad_e!(),
            "), x >= 0.0)"
        )
    };
}

impl UnaryExpr<Wgsl> for GeluTanhGradOp {
    const EXPR: &'static str = concat!(
        "select(select((",
        wgsl_gelu_tanh_grad_s!(),
        " + (2.0 * ",
        wgsl_gelu_tanh_grad_x!(),
        ") * ((",
        wgsl_gelu_tanh_grad_e!(),
        ") / ((1.0 + ",
        wgsl_gelu_tanh_grad_e!(),
        ") * (1.0 + ",
        wgsl_gelu_tanh_grad_e!(),
        "))) * 0.7978845608 * (1.0 + 0.134145 * ",
        wgsl_gelu_tanh_grad_x!(),
        " * ",
        wgsl_gelu_tanh_grad_x!(),
        ")), 1.0, x >= 40.0), 0.0, x <= -40.0)"
    );
}
impl UnaryExpr<CudaC> for GeluTanhGradOp {
    const EXPR: &'static str = c_gelu_tanh_grad_expr!();
}
impl UnaryExpr<HipC> for GeluTanhGradOp {
    const EXPR: &'static str = c_gelu_tanh_grad_expr!();
}
