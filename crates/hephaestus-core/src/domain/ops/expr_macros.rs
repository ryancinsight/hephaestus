//! Shared WGSL expression fragments for the unary markers.
//!
//! The `log1p` polynomial core below is musl's `log1pf` k = 0 path
//! (`Lg1..Lg4` with the published `2^-34.24` approximation bound), valid for
//! `1 + x` in `[sqrt(2)/2, sqrt(2)]`. One op composer instantiates it:
//! `wgsl_log1p_expr` in [`super::unary_math`] (all four bands).
//!
//! The `erfc` core below is the Numerical Recipes `erfcc` Chebyshev form
//! (`t·exp(-z²-1.26551223+t·P(t))`, `t = 1/(1+z/2)`, fractional error below
//! `1.2e-7` everywhere per §6.2), valid for `z >= 0`. Three op composers
//! instantiate it in [`super::activation`]: the `erfc` rendering and the
//! `gelu`/`gelu-grad` error-function terms (which share one argument and
//! hence one composer).
//!
//! Macro arguments are fully parenthesized literals, as the Bessel macros
//! require, since the cores splice them under `/` and `*`.

macro_rules! wgsl_l1p_s {
    ($f:literal) => {
        concat!("(", $f, " / (2.0 + ", $f, "))")
    };
}

macro_rules! wgsl_l1p_z {
    ($f:literal) => {
        concat!("(", wgsl_l1p_s!($f), " * ", wgsl_l1p_s!($f), ")")
    };
}

macro_rules! wgsl_l1p_w {
    ($f:literal) => {
        concat!("(", wgsl_l1p_z!($f), " * ", wgsl_l1p_z!($f), ")")
    };
}

macro_rules! wgsl_l1p_t1 {
    ($f:literal) => {
        concat!(
            "(",
            wgsl_l1p_w!($f),
            " * (0.40000972152 + ",
            wgsl_l1p_w!($f),
            " * 0.24279078841))"
        )
    };
}

macro_rules! wgsl_l1p_t2 {
    ($f:literal) => {
        concat!(
            "(",
            wgsl_l1p_z!($f),
            " * (0.66666662693 + ",
            wgsl_l1p_w!($f),
            " * 0.28498786688))"
        )
    };
}

macro_rules! wgsl_l1p_hfsq {
    ($f:literal) => {
        concat!("(0.5 * ", $f, " * ", $f, ")")
    };
}

macro_rules! wgsl_l1p_core {
    ($f:literal) => {
        concat!(
            "(",
            wgsl_l1p_s!($f),
            " * (",
            wgsl_l1p_hfsq!($f),
            " + (",
            wgsl_l1p_t2!($f),
            " + ",
            wgsl_l1p_t1!($f),
            ")) - ",
            wgsl_l1p_hfsq!($f),
            " + ",
            $f,
            ")"
        )
    };
}

macro_rules! wgsl_erfc_nr_t {
    ($z:literal) => {
        concat!("(1.0 / (1.0 + 0.5 * (", $z, ")))")
    };
}

// Shared exponent correction for erfc(z) and exp(z*z)*erfc(z), z >= 0.
macro_rules! wgsl_erfc_nr_correction {
    ($z:literal) => {
        concat!(
            "- 1.26551223 + ",
            wgsl_erfc_nr_t!($z),
            " * (1.00002368 + ",
            wgsl_erfc_nr_t!($z),
            " * (0.37409196 + ",
            wgsl_erfc_nr_t!($z),
            " * (0.09678418 + ",
            wgsl_erfc_nr_t!($z),
            " * (-0.18628806 + ",
            wgsl_erfc_nr_t!($z),
            " * (0.27886807 + ",
            wgsl_erfc_nr_t!($z),
            " * (-1.13520398 + ",
            wgsl_erfc_nr_t!($z),
            " * (1.48851587 + ",
            wgsl_erfc_nr_t!($z),
            " * (-0.82215223 + ",
            wgsl_erfc_nr_t!($z),
            " * 0.17087277",
            "))))))))"
        )
    };
}

macro_rules! wgsl_erfc_nr_pos {
    ($z:literal) => {
        concat!(
            "(",
            wgsl_erfc_nr_t!($z),
            " * exp(-(",
            $z,
            ") * (",
            $z,
            ") ",
            wgsl_erfc_nr_correction!($z),
            "))"
        )
    };
}

macro_rules! wgsl_erfcx_nr_pos {
    ($z:literal) => {
        concat!(
            "(",
            wgsl_erfc_nr_t!($z),
            " * exp(",
            wgsl_erfc_nr_correction!($z),
            "))"
        )
    };
}
