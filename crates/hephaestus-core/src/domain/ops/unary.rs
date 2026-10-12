//! Unary operation markers.

/// Exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpOp;

/// Fused negated exponential `exp(-x)` marker (e.g. Beer–Lambert
/// transmission).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpNegOp;

/// Natural logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LnOp;

/// Sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinOp;

/// Cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosOp;

/// Square-root operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SqrtOp;

/// Absolute value operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AbsOp;

/// Negation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct NegOp;

/// Reciprocal operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecipOp;

/// Identity/copy operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityOp;

/// Tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanOp;

/// Arcsine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsinOp;

/// Arccosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcosOp;

/// Arctangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtanOp;

/// Hyperbolic sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinhOp;

/// Hyperbolic cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoshOp;

/// Base-two logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log2Op;

/// Base-ten logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log10Op;

/// Base-two exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exp2Op;

/// Inverse hyperbolic tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtanhOp;

/// Inverse hyperbolic sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsinhOp;

/// Inverse hyperbolic cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcoshOp;

/// Exponential-minus-one operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Expm1Op;

/// Logarithm-of-one-plus operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log1pOp;

/// Sign operation marker, returning `-1`, `0`, or `1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignOp;

/// Floor operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct FloorOp;

/// Ceiling operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CeilOp;

/// Round-to-nearest-even operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundOp;

/// Truncation-toward-zero operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TruncOp;

/// Gauss error function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErfOp;

/// Complementary Gauss error function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErfcOp;

/// Natural logarithm of the absolute gamma function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LgammaOp;

/// Unnormalized `sinc` operation marker: `sin(x)/x` with `sinc(0) = 1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SincOp;

/// Bessel function of the first kind J0 operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct J0Op;

/// Bessel function of the first kind J1 operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct J1Op;

/// Modified Bessel function of the second kind K0 operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct K0Op;

/// Exact Gaussian Error Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluOp;

/// Exact Gaussian Error Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluGradOp;

/// Rectified linear unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluOp;

/// Rectified linear unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluGradOp;

/// Logistic sigmoid operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidOp;

/// Logistic sigmoid gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidGradOp;

/// Hyperbolic tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhOp;

/// Hyperbolic tangent gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhGradOp;

/// Tanh-approximated Gaussian Error Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhOp;

/// Tanh-approximated Gaussian Error Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhGradOp;

/// Sigmoid Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluOp;

/// Sigmoid Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluGradOp;

/// Softplus operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusOp;

/// Softplus gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusGradOp;

/// Mish activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MishOp;

/// Mish activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MishGradOp;

/// Exponential linear unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluOp;

/// Exponential linear unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluGradOp;

/// Hard sigmoid activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardsigmoidOp;

/// Hard sigmoid activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardsigmoidGradOp;

/// Hard swish activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardswishOp;

/// Hard swish activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardswishGradOp;

/// Softsign activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftsignOp;

/// Softsign activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftsignGradOp;
