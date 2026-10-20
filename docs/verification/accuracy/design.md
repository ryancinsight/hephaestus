# Bounded WGSL gradient evaluation and restored normal tails

This patch addresses finite evaluation and part of the coverage findings on
Hephaestus 0.26.0, inherited unchanged by the merged 0.27.0 numerical paths. It preserves public traits, value functions, C/HIP
expressions, backend requirements and every existing ULP/absolute target.
It provides no device or full portability acceptance by itself. See the
[checklist](checklist.md) and [backlog](../../../backlog.md#heph-kernel-tail-accuracy).

## Finite GELU-tanh gradient

Let `a = sqrt(2/pi)`, `b = 0.044715`, `z = a(x + b*x^3)` and
`s = sigmoid(2z)`. The derivative is
`s + 2*x*a*(1 + 3*b*x^2)*s*(1-s)`.

WGSL `select` evaluates its arguments eagerly. Selecting a limit around an
already overflowing polynomial therefore does not prevent `0*infinity`.
The new expression first sets `q = clamp(x, -40, 40)`, then forms
`w = 2*a*(q+b*q^3)` and `e = exp(-abs(w))`. It evaluates the sigmoid as
`e/(1+e)` for negative x, `1/(1+e)` for nonnegative x, and its product as
`s*(1-s) = e/((1+e)*(1+e))`.

For finite input, `|q|<=40`, `q^2<=1600`, `|w|<4632`, and `0<=e<=1`.
The exp argument never becomes positive; the denominator is in `[1,4]`.
The degree-three derivative factor stays finite. All eagerly evaluated
alternatives are safe before the outer selections enforce 0 at `x<=-40`
and 1 at `x>=40`. Permitted underflow may produce zero but cannot introduce
the previous overflowing polynomial or zero-times-infinity path.

For `r=|x|>=40`, the error from the appropriate limit is bounded by
`P(r)*exp(-w(r))`, where
`P(r)=1+2*a*r*(1+3*b*r^2)` and `w(r)=2*a*(r+b*r^3)`.
At 40, `P<14000` and `w>4600`, so the error is below
`14000*exp(-4600)<1e-1990`. This decreases thereafter:
`P'/P<=3/r`, whereas `w'=2*a*(1+3*b*r^2)>3/r` on this interval.
The bound is below half a binary64 subnormal ULP, and therefore also below
the binary32 threshold. Using 40 preserves the existing f64 capability;
the smaller binary32-only saturation threshold is not imposed on f64.
NaN-input portability is not inferred from this finite-input argument.

## Tail selection and half-erfc evaluation

The old erfc cap 8.9 excludes 308,857 normal-result binary32 inputs through
`9.194548606872559`; erfc(9) and erfc(f32(9.19)) are normal. The old GELU
and gradient floor -12.5 also excludes normal values at -12.6 and -13,
where even the shared erfc intermediate is normal.

The shared selection criterion compares the f64 erfc reference at the
rounded binary32 shader argument with `f32::MIN_POSITIVE`. It keeps
nonfinite structural controls and removes subnormal inputs through the
existing helper. Erfc uses x; GELU and its gradient use the rounded
`-x*FRAC_1_SQRT_2` argument. This restores a strict superset of the former
selected tail values. It defines coverage, not an intrinsic error budget.

At -13, the erfc intermediate is about `1.22343e-38`, which is normal,
but multiplying it by 0.5 first produces a subnormal intermediate. WGSL
may flush that term even though the final derivative is about `-1.03373e-36`
and normal. Reassociate the same identity as
`0.5*(erfc(-x/sqrt(2)) + x*(2*exp(-x*x/2))/sqrt(2*pi))`.
The doubled exponential is formed before multiplying x, so this grouping
also avoids introducing a new `2*x` overflow on huge finite inputs.
Combining the normal terms before halving avoids this specific premature
flush. It preserves the expression-only operation seam.

Farther negative inputs can still produce normal final GELU/gradient
values from a subnormal erfc intermediate. That interval remains explicitly
unqualified; full coverage needs a scaled formulation or an accepted
qualification limit. References themselves do not guarantee that every
allowed intrinsic evaluation stays normal at the exact boundary.

## Portable composition finding remains open

The independent review's final-source countermodels remain evidence about
allowed arithmetic, not observed GPU failures:

| Expression/input | Existing target | Allowed-intrinsic model error |
|---|---:|---:|
| erfc, 5.746000289916992 | 128 ULP | 161.14 ULP |
| GELU, -2.858642578125 | 32 ULP | 35.71 ULP |
| softplus, -8 | 16 ULP | 18.78 ULP |
| SiLU, -6 | 16 ULP | 17.39 ULP |
| final Mish, -2.5 | 16 ULP | 1677.14 ULP |
| final Mish-gradient, -2.5 | 32 ULP | 1384.00 ULP |

The latter two use the final -2.75 handoff, not historical Mish code.
Controlled intrinsic approximations with a propagated portable budget or
explicit adapter/driver/compiler qualification require an owner/coordinator
decision. This patch preserves the strict sampled targets and removes the
contradicted erfc/GELU tail rationales; it does not settle that broader design.
[WGSL concrete accuracy requirements](https://www.w3.org/TR/2026/CRD-WGSL-20260921/#accuracy-of-concrete-floating-point-expressions).

The shipped complete final-tree WGPU 116/116 and Metal 112/112 plus
217/217 sampled passes remain valid observations. They neither prove the
restored intervals nor establish a portable composition bound. The local
directed attempt returns `AdapterUnavailable` before all 27 planned lanes;
zero execute, and no alternative acquisition route is authorized here.
