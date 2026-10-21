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
The new expression first sets `p = abs(x) >= 40` and `q = select(x, 0, p)`, then forms
`w = 2*a*(q+b*q^3)` and `e = exp(-abs(w))`. It evaluates the sigmoid as
`e/(1+e)` for negative x, `1/(1+e)` for nonnegative x, and its product as
`s*(1-s) = e/((1+e)*(1+e))`.

For finite input, `|q|<40`, `q^2<1600`, `|w|<4632`, and `0<=e<=1`.
The exp argument never becomes positive; the denominator is in `[1,4]`.
The degree-three derivative factor stays finite. All eagerly evaluated
alternatives are safe before `select(g, select(q, 1, x>=0), p)` enforces
zero on negative saturated inputs and one on positive saturated inputs.
Here g is the numerical derivative computed above. q is zero whenever p is
true; using this typed q also keeps the nested select valid for binary64. Permitted underflow may produce zero but cannot introduce
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

## Post-draft NaN control regression

Retain both generations of automatic PR #509 failures. At b79c2682,
WGPU and Metal each fail three retained NaN controls. At f3193138, the
explicit NaN-forwarding source passes WGPU's 116 contracts, but Metal
still fails three of 2,833 main-sweep inputs at the unchanged 32 ULP bound:
90 of 112 contracts pass, one fails and 21 remain unrun. CI, lockfile,
CUDA and ROCm workflows pass; their existing physical NVIDIA/AMD skips
remain skips. The failed Metal invocation does not reach this sweep's
later root/tail/finite-saturation clauses.

The f3193138 source uses `!(x == x)` around a clamp. Source review confirms
that Naga retains that predicate in IR, MSL and SPIR-V; this does not constrain
later compiler optimization. WGSL permits a finite-math assumption. A
self-comparison may then be folded, leaving a NaN-suppressing min/max clamp.
The exact Metal intermediate values and optimized machine code are unavailable;
this mechanism is an inference, not an observed Metal trace.

A native C++ optimizer witness supplies another warning. With the actual
exported expression and finite-math enabled, self-equality disappears and
ARM's finite-math `<=` selection can accept unordered inputs. Replacing
`x<=-40` with `-x>=40` is canonicalized back and does not resolve the witness.
Both unsuccessful private prototypes remain in the coordinator's evidence.
These NaN stress inputs lie outside that compiler mode's assumptions;
the observations are optimization evidence, not C++ conformance or a substitute
for Metal qualification.

The private candidate removes both dependencies. Its polynomial uses
`q = select(x, 0, abs(x)>=40)`. Its final output is
`select(g, select(q, 1, x>=0), abs(x)>=40)`.
For a finite input inside the threshold, q is exactly x and g follows the
same native arithmetic as f3193138. Outside, q is zero, all eagerly evaluated
arithmetic remains finite, and the shared predicate selects the same exact
zero/one saturation. Infinities also select q zero. Under ordered NaN
comparisons, the magnitude predicate is false and q remains NaN in the
numerical path. No clamp, explicit self-classification, separate negative
comparison, binary32-only bitcast, new intrinsic or scalar API is introduced.
The typed q in the zero arm preserves the existing binary64 capability.

Native binary32/binary64 expression models cover 16 NaN sign/class/payload
patterns, four rounding modes, gradual/per-operation flushing, three clamp
NaN profiles and honored/folded self-equality. f3193138 loses 512 of 1,536
NaN observations in that controlled model; the candidate loses none.
All 11,520 finite observations across both gradients preserve output bits;
5,760 tanh-gradient checks stay finite without invalid/overflow, and infinity
limits and the nonzero binary64 -20 tail remain. In 1,152 saturated cases,
full CPU flags differ because discarded underflow/inexact work is removed;
no universal exception-flag identity is claimed. The native optimizer stress
witness also retains NaN for the candidate at its tested inputs. These are
scoped models and optimizer observations, not a new GPU execution receipt.

Naga 30.0.1 validates/emits the eight concrete binary32 expressions to IR,
MSL 2.4 and SPIR-V. It also validates/emits the changed binary64 gradient to
IR/SPIR-V; Metal has no binary64 acceptance here. Other seven binary32
outputs remain identical. The candidate has no clamp, equality or negative
`<=` comparison in this path. Raw binary32 SPIR-V instruction count changes
from 249 to 222; this is not a timing or register-pressure claim.

The full [WGSL finite-math assumption](https://www.w3.org/TR/WGSL/#floating-point-evaluation)
still prevents a universal NaN guarantee from being inferred. This source
delta returns for independent review before push; actual backend revalidation
and the broader tail/portable/cohort/performance gates remain open. No NaN
fixture, threshold, workflow or execution profile changes.

## Tail selection and half-erfc evaluation

The old erfc cap 8.9 excludes 308,857 normal-result binary32 inputs through
`9.194548606872559`; erfc(9) and erfc(f32(9.19)) are normal. The old GELU
and gradient floor -12.5 also excludes normal values at -12.6 and -13,
where even the shared erfc intermediate is normal.

Erfc selection compares its f64 reference at the rounded binary32 input
with `f32::MIN_POSITIVE`. The accepted #509 source also applies that criterion
to the rounded `-x*FRAC_1_SQRT_2` erfc intermediate for GELU/gradient. At -13,
combining the two normal gradient terms before halving avoids the premature
flush of the subnormal half-erfc. That accepted-source change and its prior
failures/receipts remain preserved.

### Private normal-result interval correction

The intermediate criterion first excludes x = -13.003056526184082
(bits `0xc1500c85`). MPFR at 256 and 512 bits agrees with the actual f64
value functions on the precise final-normal endpoints:

| Operation | Last normal-result input | Input bits | Excluded normal inputs |
|---|---:|---|---:|
| GELU | -13.146245956420898 | `0xc1525706` | 150146 |
| GELU gradient | -13.34188175201416 | `0xc1557859` | 355285 |

The first excluded input still has a true-normal erfc at the exact argument;
rounding the shader argument excludes that one extra lane. The true erfc is
subnormal from the next input. The Gaussian exp(-x*x/2) also becomes subnormal
before the gradient's final-normal endpoint, at bits `0xc153765b`.

The a735 normal-tail correction uses x < -13. Let z = abs(x)/sqrt(2)
and r = exp(z*z)*erfc(z).
The implementation approximates r by t*exp(N(t)), where t = 1/(1+z/2)
and N is the existing Numerical Recipes erfc exponent correction. One shared
coefficient macro serves erfc and this scaled erfc; the emitted erfc
expression remains byte-identical. Before rounding, the exact identities are

```
GELU(x) = (0.5*x*r)*exp(-x*x/2)
GELU'(x) = (r/1024 + x/(512*sqrt(2*pi)))*exp(-x*x/2 + ln(512)).
```

The first Gaussian stays normal throughout GELU's normal-result tail. The
second stays normal throughout the gradient's normal-result tail; reciprocal
scaling folds into its coefficients, avoiding a final divide or a product of
two Gaussian factors that could be regrouped into a subnormal intermediate.
The finite discarded expressions remain bounded except the nonpositive
Gaussian exponent may become negative infinity after a huge squared input;
its exponential is zero. Explicit infinity guards and the existing f64
capability flags remain. No host value function, C/HIP expression, threshold,
execution profile or public API changes.

Conformance selection now uses the final f64 operation reference: zero and
normal final results plus nonfinite structural controls. Existing selected
main/root/mid/far inputs remain, and the previously excluded normal lanes
are restored. Directed tests include -13.1, gradient -13.3, the final two
normal inputs, and the next subnormal-reference input, which the selector
excludes. No reference-margin filter, clamp or bias hides the endpoint.
The actual-expression CPU model reproduces a further accepted-source
GELU witness at bits `0xc150014b`: its gradual-underflow error exceeds
256 ULP even before the old selector's first exclusion. Directed far-band
controls retain this witness, both handoff neighbors and the Gaussian's
first-subnormal transition. The new gradient may produce negative zero for
a huge finite negative input where the old expression produces positive
zero; the unchanged sweep scores both at zero ULP. The explicit negative
infinity guard still returns positive zero.
The last-normal references sit only 83.9656 (GELU) and 63.3345 (gradient)
subnormal ULP above MIN; a legal flush following an underestimated numerical
result can therefore cause a large error even when the pre-flush error is
below 256 ULP. Endpoint controls matter in addition to interior sweeps.

This is a concrete CPU/source correction under explicit arithmetic models,
not a portable intrinsic proof or device receipt. Controlled broader exp
perturbation profiles can still violate the target and flush a final-normal
endpoint. Those stress witnesses are separate from the six existing portable
countermodels, not a proof of a newly allowed intrinsic profile. Full compiler
reassociation/transcendental transformations remain unqualified. Eager WGSL
selection adds tail work; actual performance is unmeasured. Independent review
and required actual backend validation of this private delta remain pending.

## Proposed GELU far-band correction (qualification pending)

The retained hosted run `39520852311` reports the same four normal-result GELU
far-band failures in base `6f4dd9b9` and former comparison head `9f7be898`:
`0xc14fe3f4`, `0xc14905b3`, `0xc1446f7a` and `0xc13e0c2d`, at the unchanged
256 ULP bound. All four are above -13 and use the old unscaled erfc branch.
The fixed exp2-reduction source profile reproduces this failure class without
matching exact device bits or identifying a runtime JIT mechanism. Rounding z
before its quadratic exponent can amplify the input error; moving the original-x
Gaussian outside that erfc exponent avoids that path.

Proposed provider `838cf7abb156b21b521cdc9ef52b14833305cd17` selects the
existing factored GELU expression for x < -7, its existing far-band boundary.
GELU-gradient keeps x < -13 and its 512 scaling. Coefficients, other operators,
C/HIP expressions, value references, scoring and every target remain unchanged.
The four inputs and adjacent -7 controls are additive; no old fixture is removed.
The production-only and validation-based proposals share all six published
library sources and manifests. Validation retains its earlier exact-six
fixtures and diagnostics.

The renewed comparison pins provider tree `26e0a45c44f638ccdb02646297ac71dc9af8e623`
and the retained source-expanded 24-row export map. The hosted full-library
exporter must match that exact map before acquisition. Source-identity checks
must reject any additional production delta. Existing base predicates and
strict candidate rejection remain. Public #509 a735 is unchanged by this
private preparation; the proposed #510 update carries new production source
to obtain validation evidence, not release acceptance.

Author CPU models cover the new far band and directed handoffs; independent
source/model review and new actual backend/performance evidence remain required.
Historical a735 round-toward-zero handoff observations remain preserved and
cannot be transferred as unchanged outputs after this selector changes. New
bounded CPU observations establish no portable contract, native fast-math NaN
qualification, runtime JIT identity or physical GPU backing.

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

## Proposed observed-cohort disposition (pending qualification)

The existing qualification option preserves the same mathematical operations,
supported backends, native-f64 capabilities, swept domains and numeric targets.
It does not assert that every legal WGSL intrinsic approximation or compiler
transformation meets those targets. The six allowed-intrinsic CPU countermodels
remain valid and are not observed GPU failures. No hardware cohort is accepted
by this source change alone.

The six exact binary32 witness inputs are explicit edges in their existing named
sweeps: erfc-mid `0x40b7df3c` at 128 ULP, GELU `0xc036f400` at 32 ULP, softplus
`0xc1000000` and SiLU `0xc0c00000` at 16 ULP, and Mish/Mish-gradient `0xc0200000`
at 16/32 ULP. SiLU's existing dense grid already included its witness; the other
five were absent. CPU fixture regressions retain this distinction.

A separate focused canonical-WGPU test,
`intrinsic_qualification::exact_intrinsic_witness_qualification`, executes all
six through the same unary seam and records actual input/output bits, the f64
value-function reference and its bits, ULP error, unchanged target, full WGSL
expression template and actual adapter identity. It logs all six numeric
results before asserting their targets. The complete mandatory suites still
run. Additional automatic-workflow diagnostics retain successful focused-test
output and source/tree, manifest/lock hashes, locked WGPU/Naga versions, Rust
compiler, OS and exposed host-tool identities. Required-device flags prevent
an unavailable adapter from becoming a successful qualification run. Optional
local execution without an adapter records zero witnesses and is not evidence.

Native shader-compiler versions and empty driver fields are explicitly unknown
when the platform does not expose them. Rust, Naga and Xcode versions are
provenance, not substitute proof of a driver compiler version. The evidence
must identify the observed configuration without inventing missing versions.
Any accepted scope must name that observed stack and its metadata limits;
other devices, drivers and compiler modes remain unqualified by that receipt.
A future target miss remains a correctness finding at the same target.

This proposal leaves the version-scoped contract checklist item open. Closure
requires actual exact-head witness and full-suite results, independently
reviewed identity binding, and an owner/coordinator decision accepting the
observed cohort under the unchanged targets. Preparation alone supplies no device execution or performance evidence.
Performance evidence remains a separate applicable requirement. Existing toward-zero
handoff differences, native-fast NaN limitations and unmeasured hardware and
portable-composition guarantees remain unresolved.

## Existing hosted validation route

The local `AdapterUnavailable` is a known runtime outcome, not an approval
denial. It is not retried. The existing automatic draft-PR Metal workflow has
separate GitHub-hosted execution evidence; using that normal workflow requires
no new runner registration, device access, settings change or manual dispatch.
The software-Vulkan WGPU job remains software evidence. Historical Metal
logs do not identify a physical GPU model or native driver compiler version.

The [relocatable comparison](../../../.github/validation/activation/README.md)
is validation infrastructure outside ordinary Cargo target discovery. It
overlays the identical reviewed benchmark only into disposable immutable
base/head checkouts, verifies actual compiled expressions and source/lock/
feature identity, and preserves distinct optimized binaries. Both sources
execute serially in the same existing hosted Metal job after its complete
mandatory suite and focused exact-six diagnostics. Existing job/test limits,
numerical bands, domains and production expressions remain unchanged.

The procedure retains real computed-output checks, raw batches and order,
observed adapter/driver/compiler provenance, timing scope and noise. Exposed
SDK/standalone compiler versions are not substituted for an unknown runtime
Metal JIT version. Only demonstrated baseline violations may be expected;
other baseline or candidate failures require investigation. Cancellation,
timeout, metadata mismatch or absent artifacts leaves evidence incomplete.

Preparation and a green workflow alone do not establish performance or
portable numerical acceptance. Actual final-revision observations and their
limits require independent review and owner/coordinator disposition before
any merge or publication. Coeus Miri remains deferred and unverified; the
other hardware/service and consumer release holds retain their own scope.
