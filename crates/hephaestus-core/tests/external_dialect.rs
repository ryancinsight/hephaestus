//! Proof that [`KernelDialect`] is an open extension point.
//!
//! This is an integration-test crate: it links `hephaestus-core` as an
//! external dependency and therefore sees exactly the surface a downstream
//! backend crate sees. Every item below would fail to compile against a
//! sealed `KernelDialect` — the private supertrait is unnameable and
//! unimplementable from here — so the fact that this file builds is the
//! assertion.
//!
//! The dialect modelled is SYCL/oneAPI C++, a plausible next backend that
//! does not exist in this workspace. Nothing in `hephaestus-core` mentions
//! it.

use hephaestus_core::{
    CombineExpr, CumProdOp, CumSumOp, DialectScalar, IdentityToken, KernelDialect,
};

/// SYCL C++ dialect marker, defined wholly outside `hephaestus-core`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct SyclCpp;

impl KernelDialect for SyclCpp {
    const NAME: &'static str = "sycl-cpp";
    // SYCL inherits C++ IEEE-754 semantics, as CUDA C++ and HIP C++ do.
    const IEEE_SPECIAL_VALUES: bool = true;
}

// The companion vocabularies are reachable under the orphan rule because the
// dialect marker is this crate's local type, even though the scalar and the
// op markers are both foreign.
impl DialectScalar<SyclCpp> for f32 {
    const TYPE_TOKEN: &'static str = "float";
}

impl DialectScalar<SyclCpp> for u32 {
    const TYPE_TOKEN: &'static str = "unsigned int";
}

impl CombineExpr<SyclCpp> for CumSumOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl CombineExpr<SyclCpp> for CumProdOp {
    const EXPR: &'static str = "lhs * rhs";
}

impl IdentityToken<CumSumOp, SyclCpp> for f32 {
    const TOKEN: &'static str = "0.0f";
}

impl IdentityToken<CumProdOp, SyclCpp> for f32 {
    const TOKEN: &'static str = "1.0f";
}

/// `KernelDevice::Dialect` constrains its associated type by exactly one
/// bound: `KernelDialect`. Discharging that bound here — from outside
/// `hephaestus-core`, in a const context so it is checked at compile time —
/// is necessary and sufficient for an external backend crate to name
/// `type Dialect = SyclCpp` on its own device seam.
const fn admissible_as_device_dialect<L: KernelDialect>() {}
const _: () = admissible_as_device_dialect::<SyclCpp>();

/// Dialect-generic codegen, shaped like the per-backend kernel emitters, to
/// show the external dialect is *usable* and not merely nameable.
fn fold_source<L, Op, T>() -> String
where
    L: KernelDialect,
    Op: CombineExpr<L>,
    T: IdentityToken<Op, L>,
{
    format!(
        "{ty} acc = {identity}; /* {dialect} */ acc = {expr};",
        ty = <T as DialectScalar<L>>::TYPE_TOKEN,
        identity = <T as IdentityToken<Op, L>>::TOKEN,
        dialect = L::NAME,
        expr = <Op as CombineExpr<L>>::EXPR,
    )
}

/// Read the dialect contract through the trait bound rather than off the
/// concrete type, so the assertions below exercise generic dispatch instead of
/// folding to a constant.
fn contract_of<L: KernelDialect>() -> (&'static str, bool) {
    (L::NAME, L::IEEE_SPECIAL_VALUES)
}

#[test]
fn external_dialect_reports_its_own_contract() {
    assert_eq!(contract_of::<SyclCpp>(), ("sycl-cpp", true));
}

#[test]
fn each_dialect_reports_its_own_special_value_guarantee() {
    // The external dialect participates in the same contract as the built-in
    // ones: WGSL does not promise IEEE special values, the C++ family does.
    assert_eq!(contract_of::<hephaestus_core::Wgsl>(), ("wgsl", false));
    assert_eq!(contract_of::<hephaestus_core::CudaC>(), ("cuda-c", true));
    assert_eq!(contract_of::<SyclCpp>(), ("sycl-cpp", true));
}

#[test]
fn external_dialect_carries_its_own_scalar_tokens() {
    assert_eq!(<f32 as DialectScalar<SyclCpp>>::TYPE_TOKEN, "float");
    assert_eq!(<u32 as DialectScalar<SyclCpp>>::TYPE_TOKEN, "unsigned int");
}

#[test]
fn dialect_generic_codegen_instantiates_at_the_external_dialect() {
    assert_eq!(
        fold_source::<SyclCpp, CumSumOp, f32>(),
        "float acc = 0.0f; /* sycl-cpp */ acc = lhs + rhs;"
    );
    assert_eq!(
        fold_source::<SyclCpp, CumProdOp, f32>(),
        "float acc = 1.0f; /* sycl-cpp */ acc = lhs * rhs;"
    );
}

#[test]
fn external_and_builtin_dialects_share_one_generic_emitter() {
    // The same generic function, instantiated at a core-owned dialect and at
    // the external one, yields each dialect's own tokens — the emitter is
    // dialect-parameterised, not dialect-hardcoded.
    let external = fold_source::<SyclCpp, CumSumOp, f32>();
    let builtin = fold_source::<hephaestus_core::Wgsl, CumSumOp, f32>();
    assert_ne!(external, builtin);
    assert!(builtin.contains("f32 acc"));
    assert!(external.contains("float acc"));
}
