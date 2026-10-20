//! Softplus, Mish and Mish-gradient expressions.

use super::super::UnaryExpr;
use super::super::{MishGradOp, MishOp, SoftplusOp};
use crate::domain::dialect::{CudaC, Wgsl};

// Softplus stays out of the macro (KERNEL-TAIL, ADR 0061 Decision 6):
// the naive `log(1.0 + exp(x))` collapses to `0` on the negative tail
// and overflows to `inf` past `x = 88.7`. Both spellings use the stable
// `max(x, 0) + log1p(exp(-|x|))` value form. The C spelling calls the
// native `log1p` builtin with a ternary max (the `SincOp` precedent, so
// lane promotion keeps f64 served): `fabs`/`exp`/`log1p` all resolve by
// lane width. WGSL has no `log1p`, so the correction `C(t) =
// log(1+e^t)` over `t = -|x|` is a nested select of `h(t) * exp(t)`
// arms, where each `h` is a degree-8/8/10 Chebyshev interpolant of
// `C(t)/e^t` (second-kind nodes, monomial basis solved in f64) in
// Horner form: `[-2, 0]` in `t`, `[-5, -2]` in `t + 3.5`, `[-14.5,
// -5]` in `t + 9.75`, and bare `exp(t)` below `-14.5` (relative error
// `e^t/2`). The `h` form keeps every arm flat (no cancellable
// subexpression, which the toolchain folds — see the log1p note in
// `unary.rs`) and shift-only (no divisions). A dense f32 sweep of the
// exact op sequences peaks at 3.1e-7 relative (33x inside the
// clause's `1e-5`), 60x or better off the `-14.5` step. The spelling
// nests `select` with f32-graded coefficients and calls `exp`, which
// ADR 0072 refuses at f64 on Vulkan, so it reports `SUPPORTS_F64 =
// false`.
macro_rules! softplus_t {
    () => {
        "(-abs(x))"
    };
}
macro_rules! softplus_v1 {
    () => {
        concat!("(", softplus_t!(), " + 3.5)")
    };
}
macro_rules! softplus_v2 {
    () => {
        concat!("(", softplus_t!(), " + 9.75)")
    };
}
macro_rules! softplus_h0 {
    () => {
        concat!(
            "((6.93147180559945397e-1) + ",
            softplus_t!(),
            " * (",
            "(-1.93147586403548732e-1) + ",
            softplus_t!(),
            " * (",
            "(-2.84352040718936286e-2) + ",
            softplus_t!(),
            " * (",
            "(9.41862813798837600e-3) + ",
            softplus_t!(),
            " * (",
            "(2.67131131618634533e-3) + ",
            softplus_t!(),
            " * (",
            "(-8.35752342888512116e-4) + ",
            softplus_t!(),
            " * (",
            "(-4.96122577225520736e-4) + ",
            softplus_t!(),
            " * (",
            "(-8.68501153407044322e-5) + ",
            softplus_t!(),
            " * (",
            "(-5.17321760132185863e-6)",
            ")))))))))",
        )
    };
}
macro_rules! softplus_h1 {
    () => {
        concat!(
            "((9.85198547057971052e-1) + ",
            softplus_v1!(),
            " * (",
            "(-1.45109350049478381e-2) + ",
            softplus_v1!(),
            " * (",
            "(-6.97112657087159762e-3) + ",
            softplus_v1!(),
            " * (",
            "(-2.13968209547450976e-3) + ",
            softplus_v1!(),
            " * (",
            "(-4.48016097503929817e-4) + ",
            softplus_v1!(),
            " * (",
            "(-5.84704447694812013e-5) + ",
            softplus_v1!(),
            " * (",
            "(-9.19991568228865382e-8) + ",
            softplus_v1!(),
            " * (",
            "(2.80316537311396042e-6) + ",
            softplus_v1!(),
            " * (",
            "(6.92814640194401173e-7)",
            ")))))))))",
        )
    };
}
macro_rules! softplus_h2 {
    () => {
        concat!(
            "((9.99970853800730142e-1) + ",
            softplus_v2!(),
            " * (",
            "(-2.92392498687087273e-5) + ",
            softplus_v2!(),
            " * (",
            "(-1.45769054776144263e-5) + ",
            softplus_v2!(),
            " * (",
            "(-4.78557294927118903e-6) + ",
            softplus_v2!(),
            " * (",
            "(-1.20954778997214633e-6) + ",
            softplus_v2!(),
            " * (",
            "(-2.57862383315131484e-7) + ",
            softplus_v2!(),
            " * (",
            "(-4.12964752589472005e-8) + ",
            softplus_v2!(),
            " * (",
            "(-4.39292646187302683e-9) + ",
            softplus_v2!(),
            " * (",
            "(-6.31587298884327074e-10) + ",
            softplus_v2!(),
            " * (",
            "(-1.31500316555508508e-10) + ",
            softplus_v2!(),
            " * (",
            "(-1.11694139559869794e-11)",
            ")))))))))))",
        )
    };
}
macro_rules! wgsl_softplus_expr {
    () => {
        concat!(
            "max(x, 0.0) + select(",
            "exp(",
            softplus_t!(),
            ")",
            ", select(",
            softplus_h2!(),
            " * exp(",
            softplus_t!(),
            ")",
            ", select(",
            softplus_h1!(),
            " * exp(",
            softplus_t!(),
            ")",
            ", ",
            softplus_h0!(),
            " * exp(",
            softplus_t!(),
            ")",
            ", ",
            softplus_t!(),
            " > -2.0), ",
            softplus_t!(),
            " > -5.0), ",
            softplus_t!(),
            " > -14.5)",
        )
    };
}
impl UnaryExpr<Wgsl> for SoftplusOp {
    const EXPR: &'static str = wgsl_softplus_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for SoftplusOp {
    const EXPR: &'static str = "((x) > 0.0f ? (x) : 0.0f) + log1p(exp(-fabs(x)))";
}

// Mish stays out of the macro (KERNEL-TAIL): the naive
// `tanh(log(1.0 + exp(x)))` inner transcription inherits softplus's
// negative-tail collapse (`1.0 + exp(x)` rounds to `1.0`, so the inner
// factor is `tanh(0) = 0` and the product `0`). The C spelling calls
// the native `log1p` builtin under the `tanh` (unsuffixed, so lane
// promotion keeps f64 served); no `max` gate is needed because `inf`
// flows through `tanh` to `1` and the output is `x` itself where
// softplus overflows. The WGSL spelling wraps the whole `softplus`
// value fragment in `x * tanh(...)`, adding zero new coefficients, and
// reports `SUPPORTS_F64 = false` per the `SoftplusOp` precedent
// (nested selects, f32-graded arms). `tanh` is unreliable across
// drivers at both extremes — of a tiny input (Metal returns exactly
// `0`; lavapipe returned `2^-25` for `9.35e-14` — the suite's first
// tiny input to `tanh`, which has no tiny-value coverage) and of a
// large one (Apple's unguarded `tanh` forms `inf/inf = NaN` past
// `S = 44.4`) — so the spelling guards on `x` instead of trusting it
// there: at `x <= -14.5` it takes `x * exp(x)` (there the fragment
// itself is on its bare-`exp` arm, so this reuses its value while
// skipping the `tanh`; `tanh(S) = S` to `1e-13` there and the arms
// agree to `2.5e-7` at the boundary), and at `x > 9` it takes `x`
// itself (`tanh(S)` rounds to exactly `1` past `S = 8.66`, and
// `S(x) > x`, so both arms agree exactly at the boundary). All
// `select` arms evaluate (the discarded `NaN`/`inf` never trap — the
// `SincOp` precedent, proven on Metal by the `sinc(0)` clause — and
// `select` is a real conditional move, never arithmetized). Accuracy
// is the slice-2 fragment proof plus `exp`'s own ULP, inside the
// clause's `1e-5`.
impl UnaryExpr<Wgsl> for MishOp {
    const EXPR: &'static str = concat!(
        "select(select(select(x * exp(x), x * tanh(",
        wgsl_softplus_expr!(),
        "), x > -14.5), x, x > 9.0), -0.0, (x) < -3.402823466e+38)"
    );
    const SUPPORTS_F64: bool = false;
}
// The C spelling lives in `stable.rs`: a macro-built stabilized form.

// Mish gradient (KERNEL-TAIL): the naive `tanh(log(1 + exp(x)))` lead
// term reads `0` on the negative tail, missing `~3%` of the total at
// `x = -30` — and feeding the tiny `sp` to `tanh` is exactly the
// driver hazard the `MishOp` guards dodge (Metal returns `0`,
// lavapipe garbage). The C spelling evaluates the stable
// `max + log1p` softplus under the `tanh` (CUDA/ROCm `tanh` is proven
// at tiny inputs by the `mish` clause). The WGSL spelling guards on
// `x`: at `x <= -14.5` it takes `exp(x)·(1 + x)` directly (there the
// fragment is on its bare-`exp` arm and `tanh(sp) = sp` to `1e-13`,
// so this is the composition with no `tanh` at all); on
// `(-14.5, -7]` it takes `sp + x·(1-sp²)·σ(x)` with the fragment value
// (`tanh(sp) = sp` to `2.7e-7`, `30x` inside the bound); above `-7` the
// full `tanh` form (`sp > 9e-4`, safe on every probed driver). No
// `tanh` text appears below the top arm, so no driver ever evaluates
// it on a tiny input. Nested selects report `SUPPORTS_F64 = false` per
// the `MishOp` precedent.
impl UnaryExpr<Wgsl> for MishGradOp {
    const EXPR: &'static str = concat!(
        "select(select(select(exp(x) * (1.0 + x), select(",
        wgsl_softplus_expr!(),
        // Parenthesized: the fragment is `max + select(...)`, which is
        // only `+`-safe — under `*` the bare form re-associates.
        " + x * (1.0 - (",
        wgsl_softplus_expr!(),
        ") * (",
        wgsl_softplus_expr!(),
        ")) * (1.0 / (1.0 + exp(-x))), tanh(",
        wgsl_softplus_expr!(),
        ") + x * (1.0 - tanh(",
        wgsl_softplus_expr!(),
        ") * tanh(",
        wgsl_softplus_expr!(),
        ")) * (1.0 / (1.0 + exp(-x))), x > -7.0), x > -14.5), 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)"
    );
    const SUPPORTS_F64: bool = false;
}
// The C spelling lives in `stable.rs`: a macro-built stabilized form.
